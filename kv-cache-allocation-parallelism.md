# KV Cache Block Allocation & P2P KV Transfer 설계

## 핵심 질문

각 parallelism rank가 자신이 담당하는 KV만큼의 block만 할당받을 수 있는가?
그리고 Ring Parallel Prefill → Tensor Parallel Decode 간 KV 전송을 어떻게 효율화하는가?

---

# 1부: KV Cache Block 할당 — TP / PCP / RP 비교

## Tensor Parallel (TP) — 가능, 낭비 없음

TP는 KV cache를 **head 차원**으로 분할한다.

```python
# vllm/config/model.py
def get_num_kv_heads(self, parallel_config):
    return max(1, total_num_kv_heads // parallel_config.tensor_parallel_size)
```

- 각 rank의 KV cache shape: `[num_blocks, block_size, num_kv_heads // tp_size, head_dim]`
- **model load 시점에 고정**. seq_len, request와 무관.
- 전체 tp rank를 합치면 정확히 full KV 1벌.

**왜 깔끔한가**: head 분할은 token → rank 매핑이 없다. 각 rank는 모든 token에 대해
자기 담당 head만 저장한다. block 구조와 완전히 직교(orthogonal)하므로
block allocator가 신경 쓸 필요가 없다.

---

## Prefill Context Parallel (PCP / Ulysses) — 가능, 낭비 없음

PCP는 **interleaved** 방식으로 token을 rank에 배정한다.

```
cp_world_size=2, interleave_size=1 일 때:
rank 0: token 0, 2, 4, 6, ...
rank 1: token 1, 3, 5, 7, ...
```

token i의 소유 rank = `(i / interleave_size) % cp_world_size` — **seq_len과 무관하게 고정**.

### Block 분할 방법 (virtual block)

```python
# vllm/v1/worker/block_table.py
virtual_block_size = self.block_size * total_cp_world_size

# logical block j → 모든 rank가 동일한 block ID j를 참조
# 단, 각 rank는 그 block 내에서 자기 토큰 슬롯만 사용
block_table_indices = positions // virtual_block_size
mask = (positions // interleave_size) % world_size == my_rank
```

- logical block 1개 = `world_size`개의 physical slot을 묶은 단위
- scheduler 레벨에서 `block_size *= cp_world_size`로 팽창
- 따라서 N token 처리에 필요한 block 수 = `N / (block_size * cp_world_size)`
- **각 rank는 1/cp_world_size 만큼의 physical block만 보유**

### Attention 알고리즘 (Ulysses)

interleaved 저장이 가능한 이유는 attention 방식 때문이다.

```
각 rank: scattered token 보유 → attention 전 K,V를 all-gather
각 rank: local Q × full K,V 계산
```

K,V를 어차피 전부 모으기 때문에 각 rank의 local token이 비연속이어도 무방하다.

---

## Ring Parallel (RP / Zigzag Ring Attention) — 불가능

RP는 **zigzag** 방식으로 token을 rank에 배정한다.

```
ring_chunk_len = seq_len / (2 * rp_size)

rank 0: [0, chunk_len)  ∪  [(2*rp-1)*chunk_len, seq_len)   ← head + tail
rank 1: [chunk_len, 2*chunk_len) ∪ [(2*rp-2)*chunk_len, (2*rp-1)*chunk_len)
...
```

### 왜 block 분할이 불가능한가

**결정적 이유: token → rank 매핑이 seq_len에 따라 달라진다.**

```
block_size=64, token position=100 일 때:

seq_len=800  → ring_chunk_len=50  → rank 2 소유
seq_len=1600 → ring_chunk_len=100 → rank 1 소유
seq_len=3200 → ring_chunk_len=200 → rank 0 소유
```

vLLM의 block pool은 **여러 request가 공유**한다. 동일한 physical block ID가
어떤 request에서는 rank 0의 토큰을, 다른 request에서는 rank 2의 토큰을 담게 된다.
"이 block은 rank i 전용" 이라는 정적 할당이 불가능하다.

**부가 문제: 비연속 소유 구간**

rank 0이 담당하는 block은 head 구간(앞쪽)과 tail 구간(뒤쪽)으로 쪼개진다.
DCP처럼 "logical block = 연속된 물리 블록들의 묶음"으로 표현할 수 없다.

**prefix caching과의 충돌**

같은 prefix라도 이 request의 seq_len에 따라 다른 rank가 해당 block을 "소유"하게 되므로
prefix cache hit 판정이 불가능해진다.

### Attention 알고리즘 (Ring Attention)

zigzag가 필요한 이유:

```
step 0: local Q[i] × local K[i],V[i]
step 1: 이웃 rank에서 K,V chunk 수신 → Q[i] × K[j],V[j] 누적
...
```

ring을 돌면서 K,V 조각을 **순서대로** 받아 누적해야 하므로
각 rank의 local token이 **연속된 구간**이어야 한다.
interleaved 저장처럼 scattered token을 가지면 ring pass 자체가 불가능하다.

zigzag는 causal attention의 load imbalance 문제(뒤쪽 token일수록 attend할 토큰이 많음)를
각 rank에 head 구간과 tail 구간을 동시에 배정함으로써 해결한 것이다.

---

## 요약 비교

| | TP | PCP (Ulysses) | RP (Ring Attn) |
|---|---|---|---|
| 분할 축 | head | token (interleaved) | token (zigzag) |
| token→rank 매핑 | 해당 없음 | 고정 (`i % world_size`) | seq_len 의존적 |
| block 분할 가능 | O (head 수로 shape 결정) | O (virtual block trick) | X |
| 각 rank block 수 | full (head만 줄어듦) | `1/cp_world_size` | full (낭비) |
| attention 통신 | 없음 (head 독립) | All-gather K,V per layer | Ring P2P per step |
| head 수 제약 | `num_heads % tp == 0` | `num_heads % cp == 0` | 없음 |

### RP의 현실적 처우

현재 vLLM에서 RP는 block allocator 레벨의 지원 없이 모든 rank가 full size KV cache를
할당받는다. Prefill 완료 후 각 rank는 자신의 zigzag slice만 채워진 상태이며,
decode worker로의 KV 전송 전에 all-reduce(SUM)로 모든 rank에 완전한 KV를 복제한다.

RP에서 메모리 낭비를 없애려면 zigzag 대신 contiguous split으로 바꾸는 것이 가장 단순하지만,
그 경우 causal attention의 load imbalance가 생기고 ring attention의 존재 의의가 사라진다.

---

## RP에서 load balance와 local block 할당을 동시에 달성하는 방법

### 문제의 근원

현재 `ring_chunk_len = seq_len / (2 * rp_size)` 이므로 token i의 소유 rank가
request마다 달라진다. 이것이 정적 block 할당을 막는 유일한 원인이다.

### 해법: chunk_len을 seq_len과 분리

`ring_chunk_len`을 **고정값**으로 만들면 된다:

```
fixed_chunk_len = max_model_len / (2 * rp_size)

rank 0 소유: [0, fixed_chunk_len)  ∪  [(2*rp-1)*fixed_chunk_len, max_model_len)
rank 1 소유: [fixed_chunk_len, 2*fixed_chunk_len)  ∪  [(2*rp-2)*fixed_chunk_len, ...)
...
```

token i → 소유 rank 매핑이 **seq_len과 무관하게 고정**된다.

- 각 rank는 자기 구간에 해당하는 block만 pool에서 할당받으면 됨
- rank당 block 수 = `max_model_len / (rp_size * block_size)` (1/rp_size)
- prefix caching 정상 동작: 같은 prefix는 항상 같은 rank 소유

### Computation과 Storage의 분리

**KV cache storage 매핑**: fixed_chunk_len 기반 (seq_len 무관, 정적)

**Ring attention computation**: actual seq_len 기반 zigzag 유지 (load balance 보존)

두 가지는 독립적이다. model runner가 actual seq_len으로 어느 token을 처리할지
결정하는 것과, 그 token의 KV가 어느 block에 저장될지는 별개의 문제다.
짧은 sequence에서 fixed_chunk 경계와 actual zigzag 경계가 약간 어긋날 수 있지만
block 낭비는 없어진다.

### 필요한 엔지니어링

1. **Block pool 파티셔닝**: 스케줄러가 block 할당 시 token position → rank 매핑 인식,
   PCP의 virtual block trick과 동일한 구조를 RP에 적용
2. **`max_memory_usage_bytes` 수정**: `rp_size`로 나누기 추가
3. **Slot mapping 수정**: `block_table.py`에서 fixed_chunk 기반 소유권 처리
4. **Ring attention model runner**: computation은 actual seq_len 기반 zigzag 유지

PCP가 이 구조를 이미 완성해 놓았으므로, PCP의 block partitioning 코드를
RP에 맞게 이식하는 것이 현실적인 경로다. 불가능하지 않고 구현 난이도의 문제다.

---

# 2부: P2P KV Transfer — Ring Parallel Prefill → Tensor Parallel Decode

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
