# P2P KV Transfer: Ring Parallel Prefill → Tensor Parallel Decode

Merge commit `4b595a6` (PR #11 `kv-transfer-no-ag`), 2 commits:

- `b91cbe6` — feat: Transfer kv with nixl without all-gather
- `b9b9c5b` — work with cache hit

---

## 기존 방식 (All-Gather to RP Rank 0)

Ring parallel로 prefill을 수행하면 각 RP rank는 sequence의 zigzag slice에 해당하는 KV만 보유한다.

```
rp_size=4, ring_chunk_len = seq_len / 8

rank 0:  tokens [0, L/8)  ∪ [7L/8, L)
rank 1:  tokens [L/8, 2L/8) ∪ [6L/8, 7L/8)
rank 2:  tokens [2L/8, 3L/8) ∪ [5L/8, 6L/8)
rank 3:  tokens [3L/8, 4L/8) ∪ [4L/8, 5L/8)
```

기존 방식:

1. **RP rank 0만** handshake metadata 노출 (`rp_rank != 0` 이면 `None` 반환)
2. Prefill 완료 후 `_allgather_rp_kv`로 모든 rank의 KV를 **rank 0으로 gather**,
   나머지 rank들의 데이터를 rank 0의 KV cache에 덮어씀
3. Decode TP workers는 **오직 rank 0에만** NIXL 연결, 전체 KV를 읽어감

**문제**: RP rank 0에 NIC·HBM 부하 집중. 나머지 RP ranks 유휴.

---

## 새로운 방식 (P2P: Decode → 각 RP Rank에서 직접 Read)

### 핵심 아이디어

1. 모든 RP rank가 handshake metadata를 노출하고 NIXL 엔드포인트로 동작
2. `_allgather_rp_kv`를 **SUM all-reduce**로 바꿔 **모든 RP rank에 완전한 KV를 복제**
3. Decode worker는 각 block을 "첫 토큰 소유자" RP rank에서 읽음 → read 부하 분산

---

## 블록 경계 문제와 All-Reduce 해법

### 문제

Ring chunk 경계와 KV block 경계가 맞지 않으면 하나의 block에 여러 RP rank 소유 토큰이 혼재한다.

```
block_size=64, ring_chunk_len=50

Block 0 (token  0~63):  rank0 소유(0~49)  + rank1 소유(50~63)
Block 1 (token 64~127): rank0 소유(100~127) + rank1 소유(64~99)   ← zigzag tail
```

이 상태에서 decode가 어느 rank에서 읽어도 불완전한 KV를 받게 된다.

### 해법: All-Reduce (SUM) across RP group

`_allgather_rp_kv` 재설계:

```python
# 각 RP rank:
# 1. 자신이 소유하지 않는 token 위치를 0으로 마스킹
selected = cache[block_ids]           # copy
selected.masked_fill_(~owned_mask, 0)

# 2. all_reduce(SUM) → 각 rank의 부분 KV가 합산
torch.distributed.all_reduce(selected, op=SUM, group=rp_group)

# 3. 결과를 자신의 KV cache에 씀 (모든 rank에 완전한 데이터)
cache[block_ids] = selected
```

이후 decode는 block당 특정 RP rank에서만 읽으면 올바른 KV를 얻는다.

> **비고**: 기존 "rank 0에만 gather"와 달리 모든 RP rank에 데이터 복제 → 메모리 오버헤드 있지만 read 부하 분산 가능.

### 마스크 계산 — chunk-relative 좌표계

`gpu_model_runner.py`는 ring_chunk_len을 **현재 chunk의 token 수** 기준으로 계산한다.
`_allgather_rp_kv`도 이에 맞춰 chunk-relative 좌표계를 사용:

```python
chunk_start_token  = block_idx_offset * block_size
chunk_len          = min(seq_len - chunk_start_token, num_blocks * block_size)
ring_chunk_len     = (chunk_len + rp_align - 1) // rp_align   # rp_align = 2 * rp_size

# Owned intervals — [0, chunk_len) 기준
head_start_rel = ring_chunk_len * rank
tail_start_rel = ring_chunk_len * (2 * rp_size - 1 - rank)

# Block i는 chunk-relative tokens [i*block_size, (i+1)*block_size) 담당
# global offset(block_idx_offset)은 양쪽에서 상쇄됨
```

---

## Block-to-RP-Rank 매핑 (`compute_rp_block_mapping`)

각 block의 **첫 토큰**이 속하는 RP rank에 block을 배정 → block당 정확히 하나의 읽기 소스:

```python
half = rp_size * ring_chunk_len
for block_idx in range(num_blocks):
    first_token = block_idx * block_size
    chunk_idx   = first_token // ring_chunk_len
    if first_token < half:
        rp_rank = chunk_idx                    # head region
    else:
        rp_rank = 2 * rp_size - 1 - chunk_idx # tail region (zigzag mirror)
    mapping[rp_rank].append(block_idx)
```

all-reduce 완료 후 모든 rank에 완전한 데이터가 있으므로 어느 rank에서 읽어도 무방하지만,
이 매핑으로 읽기 요청을 분산한다.

---

## 핸드셰이크 변경

### Prefill side (`gpu_worker.py`)

Composite key `rp_rank * tp_size + tp_rank`로 메타데이터 서빙:

```python
key = rp_rank * tp_size + tp_rank
return {key: metadata}   # 이전: {tp_rank: metadata}
```

### Decode side (handshake loop)

```python
for rp_rank in range(remote_rp_size):
    composite_key = rp_rank * remote_tp_size + p_remote_tp_rank
    sock.send(msgpack.encode((GET_META_MSG, composite_key)))
    metadata = decoder.decode(sock.recv())
    add_remote_agent(metadata, ..., rp_rank=rp_rank)
```

모든 RP rank와 NIXL agent 등록, `dst_xfer_side_handles[engine_id][rp_rank]`에 저장.

---

## Read 경로 (`_read_blocks_for_req`)

```
non-RP (rp_size=1):
  → 기존과 동일, rank 0에서 전체 read

RP (rp_size > 1), full cache hit (local_blocks 없음):
  → 모든 RP rank agent에 completion notification 전송 후 반환

RP (rp_size > 1), partial / no cache hit:
  1. compute_rp_block_mapping으로 block → rp_rank 매핑
  2. block_offset = num_remote - num_local  (partial hit 처리)
  3. 각 rp_rank별로:
     - fetch_indices = block_indices[block_indices >= block_offset]
     - len(fetch_indices)==0: 이미 캐시됨 → notification만 전송
     - else: _read_blocks(rp_remote_blocks, rp_local_blocks, rp_rank=rp_rank)
```

---

## Chunked Prefill 지원

두 번째 청크부터는 scheduler의 waiting→running 전환 경로를 거치지 않아
`update_state_after_alloc`이 호출되지 않는 문제 수정:

**`vllm/v1/core/sched/scheduler.py`**: running request 스케줄 루프 내에서 명시 호출

```python
if self.connector is not None:
    self.connector.update_state_after_alloc(request, new_blocks, 0)
```

**`ReqMeta.block_token_offset`**: 현재 청크 시작 token offset.
`block_idx_offset = block_token_offset // block_size`로 변환하여
`_allgather_rp_kv`에 전달 → 청크별로 올바른 chunk-relative 구간 계산.

---

## 자료구조 변경 요약

| 항목 | Before | After |
|---|---|---|
| `dst_xfer_side_handles` | `dict[EngineId, int]` | `dict[EngineId, dict[int, int]]` |
| `dst_num_blocks` key | `EngineId` | `(EngineId, rp_rank)` |
| `_remote_agents` 2nd key | `tp_rank` | `rp_rank` |
| `_rp_rank` dict | 존재 | 제거 |
| `ReqMeta.rp_rank` 필드 | 존재 (항상 0) | 제거 |
| `ReqMeta.block_token_offset` | 없음 | 추가 |

---

## 변경 파일

| 파일 | 변경 내용 |
|---|---|
| `nixl_connector.py` | P2P 핸드셰이크, block mapping, all-reduce, cache hit 처리 전반 |
| `vllm/v1/worker/gpu_worker.py` | composite key `rp_rank * tp_size + tp_rank`로 메타데이터 서빙 |
| `vllm/v1/core/sched/scheduler.py` | chunked prefill 2nd+ chunk에서 `update_state_after_alloc` 호출 |
