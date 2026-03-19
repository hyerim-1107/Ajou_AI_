# Unified4D: Track4World × SAM 3 풀 퓨전 모델 구축 보고서

> **"Segmentation IS 3D Scene Understanding"**
>
> Track4World의 4D representation을 SAM 3의 내부 표현으로 완전 통합하여,
> segmentation과 3D scene understanding이 하나의 모델에서 동시에 이루어지는 통합 아키텍처

---

## 1. 프로젝트 개요

### 1.1 목표

두 개의 오픈소스 모델을 **파이프라인이 아닌 단일 통합 모델**로 퓨전한다:

| | Track4World | SAM 3 |
|---|---|---|
| **핵심 기능** | 모든 픽셀의 3D trajectory 추적 | Open-vocabulary concept segmentation |
| **Backbone** | VGGT-style ViT (DINOv2 init, 1.2B) | Perception Encoder ViT (446M) |
| **Attention** | Frame-wise ↔ Global alternating | Windowed + Global |
| **Task Head** | Scene flow decoder (2D→3D corr.) | DETR detector + SAM2 tracker |
| **Output** | Per-pixel 3D trajectory + depth + camera | Boxes + masks + video masklets |

기존 접근: `Track4World → Grounding DINO → SAM 3` 3단계 파이프라인

Unified4D: **하나의 forward pass**에서 3D reconstruction + open-vocab segmentation + video tracking

### 1.2 핵심 가설

1. **Motion coherence = segmentation prior**: 동일한 3D motion을 공유하는 픽셀은 물리적으로 하나의 객체다
2. **Geometry-grounded detection**: 3D spatial prior가 2D bounding box보다 더 정확한 object localization을 가능하게 한다
3. **Flow-aware memory**: scene flow trajectory가 occlusion 상황에서 memory-based tracking을 강화한다

---

## 2. 아키텍처 설계

### 2.1 전체 구조

```
Input: Video frames [B, T, 3, H, W] + Text prompt

┌─────────────────────────────────────────────────┐
│ Unified Geometry-Semantic Backbone (UGSB)       │
│ 24 HybridAlternatingBlocks                      │
│ [Frame-wise Attn → Geo Cross-Attn → Global Attn]│
├─────────────────────────────────────────────────┤
│ Outputs: visual tokens + depth + pointmap       │
│          + camera poses                          │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│ Scene Flow Decoder (from Track4World)            │
│ Sparse-to-dense 2D→3D correlation               │
│ Output: per-pixel 2D flow + 3D flow + confidence│
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│ 4D Fusion Encoder (6 layers)                    │
│ Gated triple cross-attention:                    │
│   visual tokens × text tokens × flow tokens     │
│ → geometry-aware, motion-aware, semantic features│
└──────┬───────────────────┬──────────────────────┘
       │                   │
┌──────▼──────┐     ┌──────▼──────────────────────┐
│ Geo-Grounded│     │ 4D Memory Tracker            │
│ DETR Decoder│     │ SAM2 memory + flow trajectory │
│ (6 layers)  │     │ memory                       │
│ 300 queries │     │ (4 layers)                   │
└──────┬──────┘     └──────┬──────────────────────┘
       │                   │
       └─────────┬─────────┘
                 │
┌────────────────▼────────────────────────────────┐
│ Outputs:                                         │
│  • Instance masks (open-vocabulary)              │
│  • 3D bounding boxes (cx, cy, w, h, z, z_range) │
│  • Per-pixel depth map                           │
│  • Per-pixel 3D point cloud                      │
│  • Camera poses per frame                        │
│  • Dense 2D-3D scene flow                        │
│  • Motion coherence map                          │
│  • Video masklets (tracked across frames)        │
└─────────────────────────────────────────────────┘
```

### 2.2 Unified Geometry-Semantic Backbone (UGSB)

기존 두 모델의 backbone을 하나로 통합한 핵심 모듈이다.

**설계 근거**: VGGT4D 논문에서 VGGT의 global attention 레이어가 이미 implicit하게 동적 motion cue를 인코딩하고 있음이 밝혀졌다. 이는 3D geometry와 visual semantics가 동일한 representation space에서 학습될 수 있음을 의미한다.

**HybridAlternatingBlock 구조**:

각 블록은 3단계 attention을 수행한다:

1. **Frame-wise Self-Attention** (VGGT 스타일)
   - 단일 프레임 내부의 texture, appearance, local structure 인코딩
   - DINOv2의 강력한 visual feature를 활용

2. **Geometry Cross-Attention** (NEW — 퓨전의 핵심)
   - Visual tokens가 Geometry tokens (3D point cloud + depth + camera pose에서 생성)에 attend
   - 이를 통해 모든 visual token이 자신의 3D 위치를 "인식"하게 됨
   - Block 8 이후부터 활성화 (초기 레이어는 low-level features라 geometry가 아직 불안정)

3. **Global Self-Attention** (VGGT + PE 통합)
   - 모든 프레임의 모든 토큰 간 attention
   - Multi-view 3D correspondence + long-range semantic dependency 동시 캡처

**파라미터 분배**: 약 600M (backbone) — VGGT(1.2B)와 SAM3 PE(446M)의 중간 규모

### 2.3 Scene Flow Decoder

Track4World의 2D→3D correlation scheme을 그대로 채택한다.

- Backbone의 intermediate features에서 frame pair 간 correlation volume 계산
- Sparse-to-dense refinement로 dense scene flow 예측
- 출력: per-pixel `[2D flow(2) + 3D flow(3) + confidence(1)]` = 6채널

**핵심 역할**: 이 scene flow가 단순한 geometric output이 아니라, 
downstream의 Fusion Encoder에서 **segmentation을 위한 motion prior**로 직접 사용된다.

### 2.4 4D Fusion Encoder

SAM 3의 Fusion Encoder를 확장하여 scene flow conditioning을 추가한 모듈이다.

**원래 SAM 3 Fusion Encoder**: visual tokens + text tokens → cross-attention

**Unified4D Fusion Encoder**: visual tokens + text tokens + flow tokens → **gated triple cross-attention**

각 레이어에서:
1. Visual tokens에 대한 self-attention
2. Visual → Text cross-attention (semantic conditioning)
3. Visual → Flow cross-attention (motion conditioning)
4. **Gated fusion**: `gate = σ(W[text_cond; flow_cond])`로 semantic과 motion 신호를 동적으로 밸런싱

Gate의 의미:
- Gate ≈ 1: semantic 신호 우선 (정적 객체, 텍스트 프롬프트가 명확한 경우)
- Gate ≈ 0: motion 신호 우선 (빠르게 움직이는 객체, 텍스트가 모호한 경우)
- Gate ≈ 0.5: 두 신호 균등 결합 (일반적인 경우)

### 2.5 Geometry-Grounded DETR Decoder

SAM 3의 DETR decoder를 3D로 확장한다.

**핵심 변경점**:

1. **3D Positional Encoding**: Object queries에 3D 좌표 기반 positional encoding 주입.
   2D 이미지 공간이 아닌 3D 공간에서 object를 탐색한다.

2. **Dual-Signal Presence Head**: SAM 3의 presence token을 확장하여 
   motion global descriptor + text global descriptor 두 신호를 동시에 사용.
   "이 텍스트 개념이 존재하는가?" + "관련 3D motion이 관찰되는가?"

3. **3D Bounding Box**: 기존 `[cx, cy, w, h]` 대신 `[cx, cy, w, h, z_center, z_range]` 출력.
   depth 정보를 포함한 3D bounding box를 직접 예측한다.

### 2.6 4D Memory Tracker

SAM 2의 memory bank + memory attention 아키텍처를 확장한다.

**추가된 요소**:

1. **Scene Flow Memory**: 과거 프레임의 scene flow를 compact representation으로 저장.
   객체별 3D motion vector history를 유지.

2. **Flow-Aware Attention Gating**: Memory attention에서 과거 memory를 읽을 때,
   3D trajectory consistency를 체크하여 inconsistent한 memory를 down-weight.
   → Occlusion 후 재등장 시 올바른 identity 유지

3. **Motion-Aware Temporal Position Encoding**: 단순 시간 순서가 아닌,
   3D motion magnitude를 반영한 temporal position encoding.

---

## 3. Weight Initialization 전략

### 3.1 4단계 Progressive Loading

| 단계 | 모듈 | 소스 | 방법 |
|------|------|------|------|
| 1 | Backbone (frame-wise + global attn) | VGGT pretrained | Direct weight mapping |
| 2 | Backbone (geometry cross-attn) | — | Xavier initialization |
| 3 | Scene flow decoder | Track4World (deprecated) | Direct weight mapping |
| 4 | Text encoder | SAM 3 PE text encoder (deprecated) | Load & freeze |
| 5 | Fusion encoder | — | Xavier init (신규 모듈) |
| 6 | DETR decoder (base) | SAM 3 DETR (deprecated) | Partial mapping |
| 7 | DETR decoder (3D extensions) | — | Xavier init |
| 8 | Memory attention (base) | SAM 3 tracker (deprecated) | Partial mapping |
| 9 | Memory attention (flow modules) | — | Xavier init |

### 3.2 Weight Mapping 핵심

**VGGT → UGSB Backbone**:
- VGGT의 alternating attention blocks는 frame-wise와 global을 번갈아 사용
- UGSB는 하나의 블록 내에 둘 다 포함
- 매핑: `vggt.blocks[2i].attn → ugsb.blocks[i].frame_attn`, `vggt.blocks[2i+1].attn → ugsb.blocks[i].global_attn`

**SAM 3 → DETR Decoder**:
- SAM 3의 `detector.decoder.layers[i]` → `detr_decoder.layers[i]` (self-attn, cross-attn, FFN)
- Presence head: SAM 3의 단일 presence token → Unified4D의 dual-signal presence head (text branch만 초기화)

---

## 4. 학습 파이프라인

### 4.1 4단계 Progressive Training

```
Stage 1: Geometry Pretraining (2주, 8× A100)
├── 목적: 3D reconstruction 능력 확보
├── 데이터: Kubric (synthetic) + ScanNet + MegaDepth
├── Loss: depth_L1 + pointmap_L1 + camera_loss
├── Frozen: text encoder, fusion encoder, DETR decoder, memory
└── LR: 1e-4 (backbone), cosine decay

Stage 2: Flow Decoder Training (1주, 8× A100)
├── 목적: dense scene flow 예측 능력 확보
├── 데이터: Kubric + Sintel + KITTI Scene Flow
├── Loss: flow_EPE + 3D_consistency
├── Frozen: backbone (LR 1e-5), text encoder, DETR decoder, memory
└── LR: 5e-4 (flow decoder)

Stage 3: Fusion + Detection Joint Training (3주, 16× A100)
├── 목적: segmentation + 3D understanding 통합
├── 데이터: SA-Co (SAM3 데이터셋) + Kubric + DAVIS
├── Loss: focal + dice + box_L1 + depth_L1 + flow_EPE + presence_BCE
├── Frozen: text encoder
├── LR: 1e-5 (backbone), 1e-4 (fusion + DETR), 5e-5 (flow decoder)
└── Key: gated fusion이 semantic/motion balance를 자동 학습

Stage 4: Tracker Fine-tuning (1주, 8× A100)
├── 목적: video tracking with 3D trajectory consistency
├── 데이터: SA-V + DAVIS + MOSE + YouTubeVOS
├── Loss: tracking_loss + flow_consistency + identity_CE
├── Frozen: backbone, text encoder, flow decoder, DETR decoder
└── LR: 1e-4 (memory modules only)
```

### 4.2 Multi-Task Loss

```python
L_total = λ_seg × (L_focal + L_dice + L_box) 
        + λ_geo × (L_depth + L_pointmap + L_camera)
        + λ_flow × (L_EPE_2d + L_EPE_3d + L_consistency)
        + λ_pres × L_presence
        + λ_coh × L_coherence
```

기본 가중치: `λ_seg=2.0, λ_geo=1.0, λ_flow=2.0, λ_pres=1.0, λ_coh=0.5`

### 4.3 데이터 요구사항

| 데이터셋 | 용도 | 규모 | 제공 정보 |
|----------|------|------|-----------|
| Kubric | Geometry + Flow GT | 100K clips | RGB + depth + flow + camera + segmentation |
| ScanNet | Indoor 3D | 1.5K scenes | RGB + depth + camera + instance seg |
| SA-Co (SAM3) | Open-vocab concepts | 5.2M images | RGB + text + masks (4M concepts) |
| SA-V | Video segmentation | 51K videos | RGB + masklets |
| Sintel | Optical flow GT | 1K pairs | RGB + flow GT |
| KITTI SF | Scene flow GT | 200 pairs | RGB + 3D flow GT |

---

## 5. 구현 상세

### 5.1 파일 구조

```
unified4d/
├── unified4d_model.py       # 전체 모델 구현 (본 프로젝트의 핵심)
│   ├── HybridAlternatingBlock   # UGSB의 단일 블록
│   ├── GeometryEncoder          # 3D geometry → tokens
│   ├── SceneFlowEncoder         # Scene flow → tokens + coherence
│   ├── UnifiedGeometrySemanticBackbone  # 통합 backbone
│   ├── FourDFusionEncoder       # Gated triple cross-attention
│   ├── GeometryGroundedDETRDecoder  # 3D-grounded detection
│   ├── FourDMemoryBank          # Flow-aware memory bank
│   ├── FourDMemoryAttention     # Memory attention with flow gating
│   ├── Unified4D               # 메인 모델 클래스
│   └── Unified4DLoss           # Multi-task loss
├── train/
│   ├── stage1_geometry.py
│   ├── stage2_flow.py
│   ├── stage3_fusion.py
│   └── stage4_tracker.py
├── data/
│   ├── kubric_loader.py
│   ├── saco_loader.py
│   └── mixed_sampler.py
└── eval/
    ├── segmentation_eval.py
    ├── geometry_eval.py
    └── tracking_eval.py
```

### 5.2 파라미터 예산

| 모듈 | 파라미터 | 비고 |
|------|---------|------|
| UGSB Backbone | ~600M | VGGT 1.2B의 절반 규모 |
| Scene Flow Decoder | ~30M   | T4W에서 이전 (통합됨) |
| Text Encoder       | ~300M  | SAM3에서 이전, frozen (통합됨) |
| Fusion Encoder     | ~50M   | 신규 |
| DETR Decoder       | ~40M   | SAM3에서 부분 이전 (통합됨) |
| Memory Modules     | ~30M   | SAM3 tracker에서 부분 이전 (통합됨) |
| FPN + Heads | ~50M | Geometry + mask heads |
| **합계** | **~1.1B** | Trainable: ~800M (text encoder frozen) |

### 5.3 핵심 구현 포인트

**1. Geometry Cross-Attention의 점진적 활성화**

Backbone의 처음 8개 블록에서는 geometry cross-attention을 비활성화한다.
초기 레이어의 features는 low-level texture에 불과하여 3D geometry와의 cross-attention이 noise만 추가하기 때문이다. Block 8부터 features가 충분히 추상화된 후 geometry conditioning을 시작한다.

**2. Gated Fusion의 동적 밸런싱**

Fusion Encoder의 gate mechanism은 학습 과정에서 자연스럽게 task-dependent weighting을 학습한다:
- 정적 장면의 semantic segmentation → text signal 우세
- 빠르게 움직이는 객체 추적 → motion signal 우세
- 일반적 경우 → 균등 결합

이 gate는 별도의 supervision 없이 end-to-end loss만으로 학습된다.

**3. Flow-Aware Memory Gating**

Memory attention에서 과거 프레임의 memory를 읽을 때, scene flow consistency를 체크한다. 3D trajectory가 갑자기 불연속적인 memory entry는 occlusion이나 다른 객체와의 혼동을 의미하므로 attention weight를 낮춘다.

---

## 6. 평가 계획

### 6.1 벤치마크

| Task | 벤치마크 | 메트릭 | 비교 대상 |
|------|----------|--------|-----------|
| Open-vocab Seg | SA-Co Gold/Silver | cgF1, mIoU | SAM 3, DINO-X |
| Video Tracking | SA-V test | cgF1, pHOTA | SAM 3, SAM 2 |
| Scene Flow | Sintel, KITTI | EPE (2D/3D) | Track4World, RAFT-3D |
| Depth | ETH3D, NYU | AbsRel, δ<1.25 | VGGT, Depth Anything |
| 3D Tracking | Kubric | AJ, δ_avg | Track4World, CoTracker |

### 6.2 핵심 검증 시나리오

Store Guard (CCTV 기반 매장 컴플라이언스) 관점에서 중요한 시나리오:

1. **Occlusion Recovery**: 사람이 선반 뒤로 지나갈 때 마스크 유지
2. **Fast Motion**: 빠르게 걷는 고객의 정확한 segmentation
3. **Text Discrimination**: "빨간 옷 입은 사람" vs "파란 옷 입은 사람" 구별
4. **Multi-Object**: 동시에 여러 고객 + 물체 추적
5. **3D Box Accuracy**: depth 기반 3D bounding box의 실용적 정확도

### 6.3 Ablation Study 계획

| 실험 | 변경 | 기대 결과 |
|------|------|-----------|
| w/o Geometry Cross-Attn | Block 내 geo cross-attn 제거 | 3D 관련 metrics 하락 |
| w/o Flow Conditioning | Fusion encoder에서 flow branch 제거 | Occlusion tracking 하락 |
| w/o Gated Fusion | Gate를 0.5 고정 | 전반적 약간 하락 |
| w/o Dual Presence | Motion signal 제거, text only | Phantom detection 증가 |
| w/o Flow Memory | Memory bank에서 flow memory 제거 | Long-term tracking 하락 |
| Pipeline Baseline | T4W + SAM3 별도 실행 후 결합 | 느리지만 각 task에서 약간 우위 가능 |

---

## 7. 리스크 및 대응

### 7.1 기술적 리스크

| 리스크 | 심각도 | 대응 |
|--------|--------|------|
| Backbone 통합 시 양쪽 성능 동시 하락 | 높음 | Progressive training + 단계별 검증 |
| Geometry cross-attn이 초기 학습에서 발산 | 중간 | Block 8 이후 활성화 + warm-up |
| Multi-task loss 밸런싱 실패 | 중간 | Uncertainty weighting 또는 GradNorm |
| 메모리/연산 비용 초과 | 중간 | Mixed precision + gradient checkpointing |
| Scene flow GT 부족 (실세계 데이터) | 높음 | Kubric synthetic → real domain adaptation |

### 7.2 Fallback 전략

풀 퓨전이 예상보다 어려운 경우, 단계적 후퇴 전략:

1. **Level 3 (풀 퓨전)**: 이 문서에서 설계한 Unified4D → 실패 시
2. **Level 2 (Adapter 퓨전)**: SAM 3 backbone을 유지하고, T4W features를 adapter로 주입
3. **Level 1 (Late 퓨전)**: 두 모델 별도 실행, output level에서 결합

---

## 8. Store Guard 적용 시나리오

### 8.1 기존 파이프라인 vs Unified4D

**기존**: `Grounding DINO → SAM 2.1 → Tracker → Depth Anything → 규칙 기반 판정`
- 5개 모델 순차 실행, 레이턴시 높음
- 각 모델 간 정보 손실
- Depth 정보가 segmentation에 반영되지 않음

**Unified4D**: `단일 모델 → 전체 출력`
- 하나의 forward pass에서 segmentation + depth + tracking + 3D boxes
- 3D motion coherence가 segmentation 품질을 직접 개선
- CCTV 실시간 처리에 유리 (파이프라인 오버헤드 제거)

### 8.2 신규 가능 기능

1. **3D 공간 기반 컴플라이언스 체크**: 단순 2D bbox가 아닌 3D position으로 구역 침범 판단
2. **Motion-based 이상 감지**: coherence map에서 비정상적 motion pattern 탐지
3. **Text 기반 실시간 검색**: "카트를 끌고 있는 사람" 같은 자연어 쿼리로 실시간 필터링

---

## 9. 결론 및 로드맵

### 9.1 기여점 요약

1. **HybridAlternatingBlock**: VGGT와 SAM3 PE의 attention 패턴을 하나의 블록으로 통합
2. **4D Fusion Encoder**: Text + Motion gated cross-attention으로 semantic과 geometric 정보를 동적 결합
3. **Geometry-Grounded DETR**: 3D positional encoding + dual-signal presence head
4. **Flow-Aware Memory**: Scene flow trajectory memory로 occlusion-robust tracking

### 9.2 로드맵

| 기간 | 마일스톤 |
|------|----------|
| Week 1-2 | Feature visualization 실험 (CKA, PCA) — 퓨전 포인트 실증 검증 |
| Week 3-4 | UGSB Backbone 구현 + VGGT weight loading |
| Week 5-6 | Stage 1 학습 (Geometry pretraining on Kubric) |
| Week 7-8 | Stage 2 학습 (Flow decoder training) |
| Week 9-12 | Stage 3 학습 (Fusion + Detection joint training) |
| Week 13-14 | Stage 4 학습 (Tracker fine-tuning) |
| Week 15-16 | 벤치마크 평가 + Ablation study |
| Week 17-18 | Store Guard 통합 + 실제 CCTV 테스트 |

### 9.3 필요 자원

- GPU: 16× A100 80GB (Stage 3 기준, 약 3주)
- 스토리지: ~10TB (SA-Co + Kubric + video datasets)
- 전체 학습 기간: 약 7주 (순차) / 5주 (병렬 최적화 시)

---

*이 문서는 Track4World (TencentARC) × SAM 3 (Meta) 풀 퓨전 아키텍처의 설계, 구현, 학습 전략을 기술한 기술 보고서입니다.*

*모델 코드: `unified4d_model.py` (약 900 lines)*
