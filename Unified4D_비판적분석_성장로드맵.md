# Unified4D: 비판적 분석 & GitHub ★1K+ 성장 로드맵

> **작성일**: 2026-03-19  
> **프로젝트**: Track4World × SAM 3 Full Fusion Architecture  
> **문서 목적**: 근본적 문제 진단 + 오픈소스 성장 전략 수립

---

## 목차

1. [현재 프로젝트의 근본적 문제 진단](#1-근본적-문제-진단)
2. [기술적 핵심 버그 목록](#2-기술적-핵심-버그-목록)
3. [GitHub ★1K+ 달성을 위한 전략 분석](#3-github-1k-달성-전략-분석)
4. [리포지토리 재설계 계획](#4-리포지토리-재설계-계획)
5. [단계별 개발 로드맵](#5-단계별-개발-로드맵)
6. [커뮤니티 성장 전략](#6-커뮤니티-성장-전략)
7. [경쟁 프로젝트 분석 및 차별화](#7-경쟁-프로젝트-분석-및-차별화)
8. [성공 지표 및 KPI](#8-성공-지표-및-kpi)

---

## 1. 근본적 문제 진단

### 1.1 프로젝트 정체성 위기: "무엇을 만들고 있는가?"

현재 Unified4D는 세 가지 성격이 뒤섞여 있어 타겟 독자가 불명확합니다.

```
지금 이 프로젝트는:
  A) 연구 논문을 위한 실험 코드인가?
  B) 실용적인 오픈소스 라이브러리인가?
  C) Store Guard를 위한 내부 인프라인가?

→ 셋 다도 아니고, 셋 다이기도 한 상태
```

**이것이 왜 치명적인가:**

GitHub 스타를 받으려면 사람들이 "이거 내 문제를 해결해준다"는 즉각적인 인식이 필요합니다. 현재 README는 Train4World × SAM 3를 합친다는 아이디어는 흥미롭지만, 지금 당장 누군가가 `pip install`하거나 체크포인트를 다운받아서 30분 안에 결과를 볼 수 없습니다. **결과를 볼 수 없는 프로젝트는 스타를 받지 못합니다.**

레퍼런스: SAM (Meta AI)이 하루 만에 5천 스타를 받은 이유는 그날 바로 데모를 실행할 수 있었기 때문입니다.

---

### 1.2 "풀 퓨전"이라는 주장의 허구성

보고서에서 반복적으로 강조하는 **"파이프라인이 아닌 단일 통합 모델"** 이라는 주장에는 치명적인 모순이 있습니다.

**주장:** VGGT(1.2B) + SAM3 PE(446M) → 단일 600M backbone

**현실:**
```python
# UGSB의 실제 구현
self.frame_attn = nn.MultiheadAttention(1024, 16)   # VGGT 흉내
self.global_attn = nn.MultiheadAttention(1024, 16)  # SAM3 흉내
```

두 모델의 핵심 강점은 사전학습 가중치(pretrained weights)에 있습니다. VGGT의 frame-wise attention은 DINOv2로 초기화된 1.2B 파라미터의 3D geometry 이해 능력을 담고 있습니다. SAM 3의 PE는 SA-1B 10억 마스크로 학습된 semantic 표현입니다.

이 두 가중치를 `nn.MultiheadAttention` 두 개에 매핑하는 것은, 피카소와 렘브란트의 기법을 합친다면서 둘 다의 그림에서 랜덤한 픽셀을 가져와 합치는 것과 같습니다. 기법이 아닌 픽셀을 합쳤으니 의미가 없습니다.

**진짜 풀 퓨전의 조건:**
- Feature space alignment: 두 모델이 같은 공간에서 representation을 학습한다는 실증
- CKA 실험: cross-model similarity가 실제로 존재한다는 근거
- Knowledge distillation: 각 모델의 강점을 증류해서 새 모델에 이전

현재 프로젝트에는 이 중 하나도 실제로 수행되지 않았습니다.

---

### 1.3 데이터 현실 vs 설계 야망의 괴리

| 설계 요구사항 | 현실 |
|---|---|
| SA-Co 5.2M 이미지 (4TB) | Meta AI 별도 신청 필요, 수 주 소요 가능 |
| Kubric 100K clips (2TB) | Google Cloud Storage gsutil 필요, 대역폭 비용 발생 |
| 16× A100 80GB (Stage 3) | H100 80GB Colab 인스턴스 1개가 최대 현실 |
| SA-V 51K videos (1.4TB) | 별도 신청 필요 |

**결론:** 설계 스펙대로 학습하려면 AWS/GCP 기준 월 수백만 원의 GPU 비용과 수십 TB의 스토리지가 필요합니다. 현재 코드는 이 환경 없이는 절대 돌아갈 수 없습니다. 반면 GitHub에서 스타를 받으려면 누군가가 자신의 로컬 환경이나 Colab에서 실행해보고 "이거 된다"는 경험을 해야 합니다.

---

### 1.4 아키텍처 복잡도의 역설: 복잡할수록 기여자가 사라진다

현재 `unified4d_model.py` 하나가 900줄입니다. 그 안에 10개 모듈이 전부 들어가 있습니다. 이것은 두 가지 문제를 만듭니다:

**첫째, 디버깅 불가능성:** 학습 중 loss가 NaN이 나왔을 때 어느 모듈이 문제인지 파악이 극도로 어렵습니다.

**둘째, 기여 진입장벽:** 오픈소스 기여자가 "이 부분만 개선해볼게요"라고 할 때, 900줄 파일에서 어디를 수정해야 하는지 파악하는 것만으로도 수 시간이 걸립니다. PyTorch, Hugging Face 같은 성공한 오픈소스는 모두 모듈이 명확히 분리되어 있고, 각 파일이 단일 책임을 가집니다.

---

## 2. 기술적 핵심 버그 목록

### Bug #1: text_encoder 모듈 부재 (Critical)

**위치:** `unified4d_model.py`, `Unified4D` 클래스

**문제:**
```python
# 보고서에서:
# "Text Encoder ~300M (SAM3에서 이전, frozen)"
# → 근데 Unified4D 클래스 어디에도 없음

class Unified4D(nn.Module):
    def __init__(self, ...):
        self.backbone = UnifiedGeometrySemanticBackbone(...)
        self.flow_enc = SceneFlowEncoder(...)
        # self.text_encoder = ???  ← 없음
```

**결과:** Stage 3에서 CLIP 텍스트 인코딩 결과를 외부에서 주입받는데, 이것이 모델의 일부인지 외부 의존성인지 불명확합니다. 논문급 모델이 되려면 end-to-end로 텍스트 → 마스크가 가능해야 합니다.

---

### Bug #2: use_memory 플래그에서의 Tensor Shape 혼동 (Critical)

**위치:** `unified4d_model.py`, `Unified4D.forward()`

```python
if use_memory:
    frame_mem, flow_mem = self.memory_bank.get_memory_tokens()
    queries_init = fused[:, :self.detr_decoder.num_queries, :]
    # ↑ fused는 [B, T*N, d_model]이고 T*N ≈ 1369
    # num_queries = 300
    # 즉 visual token의 처음 300개를 object query로 오인하는 버그
    queries_mem = self.memory_attn(queries_init, frame_mem, flow_mem)
    fused_for_detr = fused.clone()
    fused_for_detr[:, :self.detr_decoder.num_queries, :] = queries_mem
    # ↑ visual token을 query로 덮어씀 → forward 결과 완전히 잘못됨
```

**올바른 구현:**
```python
# object queries는 detr_decoder.query_embed에서 생성되어야 함
# memory attention은 queries와 memory 사이에서 수행되어야 하며
# fused (visual feature)를 수정하는 게 아님
```

---

### Bug #3: flow_consistency loss의 단위 불일치 (High)

**위치:** `unified4d_model.py`, `Unified4DLoss.flow_consistency()`

```python
def flow_consistency(self, flow_2d, flow_3d):
    mag_2d = flow_2d.norm(dim=-1, keepdim=True)  # 픽셀/프레임 단위
    mag_3d = flow_3d.norm(dim=-1, keepdim=True)  # 미터/프레임 단위
    ratio = (mag_2d / (mag_3d + 1e-6)).clamp(0.1, 10.0)
    return (torch.log(ratio) ** 2).mean()
    # ↑ 이 loss는 비율을 1로 만들려 함
    # 근데 픽셀/미터 비율은 focal length에 따라 달라짐
    # 고정된 ratio target이 없으면 모델이 depth를 이상하게 학습함
```

---

### Bug #4: _SimplePixelDecoder의 upsampling 부족 (High)

```python
class _SimplePixelDecoder(nn.Module):
    def __init__(self, d_model, embed_dim):
        self.up1 = nn.ConvTranspose2d(d_model, d_model//2, 2, stride=2)  # 2x
        self.up2 = nn.ConvTranspose2d(d_model//2, d_model//4, 2, stride=2)  # 4x
        # patch_size=14이면 14x upsampling이 필요
        # 4x 후 F.interpolate로 때우는 건 edge quality 매우 낮음
```

SAM2의 실제 mask decoder는 2단계 upsampling + skip connection + hypernetwork를 사용합니다. 현재 구현으로는 SAM 3과의 마스크 품질 비교가 의미가 없습니다.

---

### Bug #5: Stage 3 Kubric branch에서 flow loss가 역효과 (Medium)

```python
def kubric_branch_loss(...):
    # flow_consistency만 계산 (GT 없음)
    l_cons = self.flow_consistency(preds["flow_2d"], preds["flow_3d"])
    # 위에서 지적한 단위 문제 + GT 없이 consistency만 주면
    # 모델이 flow_2d와 flow_3d를 모두 0으로 만드는 게 loss를 낮추는 쉬운 방법
```

---

### Bug #6: load_unified4d_weights() 내 이중 import (Low)

```python
def load_unified4d_weights(...):
    ...
    if t4w_ckpt_path and os.path.isfile(t4w_ckpt_path):
        import os  # ← 함수 최상단에 이미 import 되어 있어야 할 것이 함수 내부 깊숙이
```

---

## 3. GitHub ★1K+ 달성 전략 분석

### 3.1 GitHub 스타 패턴 분석: 무엇이 스타를 받는가

2020-2025년 AI/Vision 분야에서 1K+ 스타를 받은 프로젝트들의 공통점:

| 프로젝트 | 스타 수 | 핵심 요인 |
|---|---|---|
| Segment Anything (Meta) | 46K+ | 그날 데모 가능 + arXiv 동시 공개 |
| Grounding DINO | 7K+ | 기존 DETR보다 명확히 나은 결과 + demo video |
| Track Anything | 5K+ | SAM + 트래킹 조합 아이디어 간결 + 즉시 실행 |
| Depth Anything | 8K+ | SOTA 결과 + 단순한 API (`depth = model(img)`) |
| CoTracker | 3K+ | 논문 + 코드 + Colab 노트북 동시 공개 |

**공통 패턴:**
1. **즉시 실행 가능** (pip install + 5분 이내 결과)
2. **시각적으로 인상적인 결과** (GIF/동영상)
3. **단순한 API** (복잡성 숨기기)
4. **arXiv 또는 블로그와 동시 공개**
5. **Twitter/X 바이럴** (AI 커뮤니티)

현재 Unified4D는 이 중 어느 것도 충족하지 못합니다.

---

### 3.2 Unified4D의 실제 경쟁력 있는 아이디어

비판적으로 분석하더라도, 이 프로젝트에는 진짜 가치 있는 아이디어가 있습니다:

**진짜 차별화 포인트 (살려야 할 것):**

1. **Motion coherence = segmentation prior** 아이디어는 학술적으로 새롭고 실용적
2. **Flow-aware memory tracking** - occlusion robustness를 motion으로 강화하는 접근
3. **Store Guard 실사용 시나리오** - 추상적 연구가 아닌 구체적 응용
4. **4단계 progressive training** 전략 - 재현 가능한 학습 레시피

**문제는 이 아이디어들이 900줄 코드 안에 묻혀 있고, 실행할 수 없다는 것**

---

### 3.3 스타를 받기 위한 핵심 전제: Proof of Concept First

```
현재 전략:
  완전한 아키텍처 설계 → 완전한 학습 파이프라인 → 논문 → 공개
  (예상 시간: 6-12개월, 현실화 가능성: 낮음)

필요한 전략:
  작동하는 MVP → 시각적 결과 → 커뮤니티 공유 → 개선 → 논문
  (예상 시간: 4-6주, GitHub 스타 획득 가능성: 높음)
```

---

## 4. 리포지토리 재설계 계획

### 4.1 프로젝트 정체성 재정의

**새로운 포지셔닝:**

> **"Unified4D: Motion-Aware Open-Vocabulary Video Segmentation"**
>
> SAM 3의 텍스트 기반 segmentation에 Track4World의 4D motion 정보를 융합하여,
> occlusion과 빠른 움직임에 강인한 비디오 segmentation을 실현합니다.
> Level 1 (Late Fusion)부터 Level 3 (Full Fusion)까지 단계적으로 제공합니다.

이 포지셔닝의 장점:
- Level 1은 지금 당장 작동 가능 (두 모델 별도 실행 후 output 결합)
- Level 3는 장기 연구 목표
- 사용자는 즉시 가치를 얻을 수 있음

---

### 4.2 리포지토리 구조 재설계

```
unified4d/
├── README.md                    ← 30초 안에 GIF와 결과물 보여주기
├── INSTALL.md
├── examples/
│   ├── quick_start.ipynb        ← Colab 버튼 달린 15분 튜토리얼
│   ├── demo_occlusion.py        ← 핵심 킬러 데모
│   └── store_guard_demo.py
│
├── unified4d/
│   ├── __init__.py
│   ├── models/
│   │   ├── level1_late_fusion.py     ← 지금 당장 작동
│   │   ├── level2_adapter_fusion.py  ← 3개월 목표
│   │   └── level3_full_fusion.py     ← 6개월 목표
│   │
│   ├── modules/                      ← 각 파일이 단일 책임
│   │   ├── hybrid_attention.py
│   │   ├── geometry_encoder.py
│   │   ├── scene_flow_decoder.py
│   │   ├── fusion_encoder.py
│   │   ├── detr_decoder.py
│   │   ├── memory_bank.py
│   │   └── pixel_decoder.py
│   │
│   ├── data/                         ← 현재 구현 유지 + 버그 수정
│   ├── train/
│   └── eval/
│
├── scripts/
│   ├── download_checkpoints.sh
│   ├── run_demo.sh
│   └── colab_setup.sh
│
├── docs/
│   ├── architecture.md
│   ├── training_guide.md
│   └── api_reference.md
│
├── tests/                           ← 현재 없음, 반드시 추가
│   ├── test_models.py
│   ├── test_data_loaders.py
│   └── test_loss_functions.py
│
└── assets/
    ├── demo.gif                     ← 반드시 필요
    ├── architecture.png
    └── results/
```

---

### 4.3 README 재설계 (스타를 받는 README의 구조)

```markdown
# Unified4D: Motion-Aware Video Segmentation

[데모 GIF - occlusion 전후 마스크 유지 장면]

**"움직이는 것을 더 잘 분리한다"**

| | SAM 3 단독 | Track4World 단독 | Unified4D |
|---|---|---|---|
| Occlusion 후 추적 | ❌ | ⚠️ | ✅ |
| 자연어 쿼리 | ✅ | ❌ | ✅ |
| 3D 위치 추정 | ❌ | ✅ | ✅ |
| 실시간 처리 | ✅ | ⚠️ | ✅ |

## 빠른 시작 (5분)

\`\`\`bash
pip install unified4d
unified4d demo --video your_video.mp4 --prompt "person"
\`\`\`

[![Open In Colab](colab-badge.svg)](colab_link)

[자세한 문서] [논문] [데모 영상]
```

---

## 5. 단계별 개발 로드맵

### Phase 0: 기반 수정 (2주)

**목표:** 현재 코드의 치명적 버그를 수정하고 최소한 smoke test가 통과되게 만들기

#### Week 1: 버그 수정

- [ ] **Bug #2 수정** (use_memory shape 혼동)
  ```python
  # 수정: memory attention은 fused가 아닌 detr decoder queries와 수행
  class Unified4D:
      def forward(self, frames, text_tokens, ...):
          # ... backbone, flow, fusion ...
          # memory attention은 DETR decoder 내부로 이동
          det_out = self.detr_decoder(fused, text_tokens, motion_global,
                                       memory_bank=self.memory_bank if use_memory else None)
  ```

- [ ] **Bug #3 수정** (flow consistency의 projection 추가)
  ```python
  def flow_consistency(self, flow_2d, flow_3d, depth, K):
      # depth와 intrinsics로 2D flow를 3D로 lift한 뒤 비교
      flow_3d_from_2d = lift_2d_to_3d(flow_2d, depth, K)
      return F.l1_loss(flow_3d, flow_3d_from_2d)
  ```

- [ ] **Bug #4 수정** (_SimplePixelDecoder 개선)
  ```python
  # 최소한 4단계 upsampling으로 교체
  # skip connection 추가
  class PixelDecoder(nn.Module):
      def __init__(self, d_model, embed_dim):
          # up1: 2x, up2: 2x, up3: 2x, up4: 2x → 16x total
          # patch_size=14이면 interpolate 마지막에 최소화
  ```

- [ ] **text_encoder 구조 명확화**: 외부 CLIP vs 내부 모듈 결정

- [ ] **load_unified4d_weights() 리팩토링**: 실제로 동작하는 key mapping

#### Week 2: 테스트 추가 및 Level 1 구현

- [ ] `tests/` 폴더 생성 및 최소 테스트 작성
  ```python
  # tests/test_models.py
  def test_forward_pass():
      model = Unified4D(img_size=224, embed_dim=256, backbone_depth=4)
      frames = torch.randn(1, 4, 3, 224, 224)
      text = torch.randn(1, 10, 256)
      out = model(frames, text)
      assert out["masks"].shape == (1, 300, 224, 224)
      assert out["depth"].shape == (1, 4, 224, 224)
  
  def test_loss_not_nan():
      ...
  
  def test_use_memory_consistent():
      ...
  ```

- [ ] **Level 1 Late Fusion 구현** (이것이 첫 번째 배포 가능한 버전)
  ```python
  class Unified4DLevel1:
      """
      Track4World + SAM 3를 별도로 실행하고
      output level에서 motion-aware로 결합.
      지금 당장 작동 가능.
      """
      def __init__(self, t4w_model, sam3_model):
          self.t4w = t4w_model
          self.sam3 = sam3_model
      
      def forward(self, frames, text_prompt):
          # 1. Track4World로 scene flow 추출
          flow_3d, depth = self.t4w(frames)
          # 2. Motion coherence map 계산
          coherence = compute_motion_coherence(flow_3d)
          # 3. SAM 3 segmentation
          masks, boxes = self.sam3(frames, text_prompt)
          # 4. Motion prior로 마스크 refine
          masks_refined = refine_masks_with_motion(masks, coherence, flow_3d)
          return masks_refined, boxes, depth
  ```

---

### Phase 1: Proof of Concept (4-6주)

**목표:** 시각적으로 인상적인 결과물 + Colab 데모 + 첫 공개

#### Week 3-4: Level 1 검증 및 결과 생성

**핵심 실험: SAM 3 단독 vs Unified4D Level 1 비교**

```python
# 검증 시나리오 (보고서 §6.2 기반)
scenarios = [
    "occlusion_recovery",  # 선반 뒤로 사라졌다 재등장
    "fast_motion",          # 빠른 이동
    "crowded_scene",        # 여러 사람 동시 추적
]

# 핵심 메트릭
metrics = {
    "J&F":     "DAVIS 기준 tracking 품질",
    "MOTA":    "Multi-Object Tracking Accuracy", 
    "id_sw":   "Identity Switch 횟수 (낮을수록 좋음)",
    "latency": "초당 처리 프레임 수",
}
```

**결과 시각화:**
- occlusion 전후 마스크 유지 여부를 보여주는 GIF 생성
- SAM 3 단독과 나란히 비교하는 side-by-side 영상
- DAVIS/YouTube-VOS 일부 시퀀스로 정량 비교

#### Week 5-6: 데모 환경 구축 및 첫 공개

**Colab 노트북 필수 구성:**
```
1. 환경 설정 (5분)
   - pip install
   - 체크포인트 다운로드 (Hugging Face Hub 활용)

2. 기본 데모 (10분)
   - 영상 업로드
   - 텍스트 프롬프트 입력
   - 결과 시각화

3. 비교 데모 (5분)
   - SAM 3 단독 vs Unified4D Level 1 side-by-side
```

**체크포인트 배포:**
```python
# Hugging Face Hub에 올리기
# pip install huggingface_hub
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="./checkpoints",
    repo_id="your-name/unified4d",
    repo_type="model",
)
```

**첫 공개 전략:**
- GitHub + arXiv (tech report 수준이라도) 동시 공개
- Twitter/X에 데모 GIF 게시 (@AK391 같은 AI 인플루언서 태그)
- Hugging Face Spaces에 gradio 데모 배포
- Reddit r/MachineLearning, r/computervision 포스팅

---

### Phase 2: Level 2 Adapter Fusion (2-3개월)

**목표:** SAM 3 backbone을 유지하면서 Track4World features를 adapter로 주입

#### 아키텍처 설계 (Level 2)

```python
class Unified4DLevel2(nn.Module):
    """
    SAM 3 backbone을 frozen으로 유지.
    Track4World features를 lightweight adapter로 주입.
    
    학습 파라미터: ~50M (adapter만)
    전체 파라미터: ~900M (frozen backbone 포함)
    """
    
    def __init__(self, sam3_backbone, adapter_dim=64):
        super().__init__()
        # SAM 3 backbone (frozen)
        self.sam3 = sam3_backbone
        for p in self.sam3.parameters():
            p.requires_grad = False
        
        # Track4World feature extractor (frozen)  
        self.t4w_encoder = T4WEncoder()
        for p in self.t4w_encoder.parameters():
            p.requires_grad = False
        
        # Lightweight adapter (trainable, ~50M)
        self.adapters = nn.ModuleList([
            MotionAdapter(embed_dim=1024, adapter_dim=adapter_dim)
            for _ in range(8)  # SAM3의 global attention 위치에 삽입
        ])
        
        # Flow-aware memory (trainable)
        self.flow_memory = FlowAwareMemory(d_model=256)
```

**학습 전략:**
- 데이터: DAVIS + YouTube-VOS (공개 접근 가능, 수십 GB 수준)
- GPU: H100 1개로 가능 (adapter만 학습하므로)
- 기간: 2-3주 학습
- 비용: Colab Pro+ 정도로 가능

---

### Phase 3: Level 3 Full Fusion & 논문 (4-6개월)

**목표:** 진짜 풀 퓨전 모델 + 학술 논문 투고

#### 선행 조건 (Phase 3 시작 전 반드시 완료)

1. **CKA 실험 완료**: Track4World와 SAM 3 간 feature space alignment 실증
   ```python
   # discover_layers.py + extract_and_visualize.py 실제 실행
   # 결과: 어느 레이어에서 fusion이 가장 효과적인지 근거 확보
   ```

2. **Level 1, 2 결과로 충분한 baseline 확보**
   - Level 1: SAM 3보다 occlusion tracking이 X% 향상
   - Level 2: 추가 Y% 향상
   - Level 3: 추가 Z% 향상 (이것이 논문의 핵심 주장)

3. **데이터 접근권 확보**
   - Meta AI SA-Co 데이터 신청
   - VGGT 체크포인트 접근 (현재 공개됨)
   - SAM 3 체크포인트 접근 (현재 공개됨)

#### 논문 투고 타겟

| 학회/저널 | 데드라인 | 우선순위 |
|---|---|---|
| ICCV 2025 | 2025년 3월 | ★★★★★ |
| NeurIPS 2025 | 2025년 5월 | ★★★★ |
| CVPR 2026 | 2025년 11월 | ★★★★★ |
| ECCV 2026 | 2026년 3월 | ★★★ |

---

### Phase 4: 에코시스템 구축 (지속적)

**목표:** 단순한 코드 공개를 넘어 커뮤니티가 기여하는 프로젝트로 성장

#### 필수 인프라

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: pytest tests/ -v
      - name: Check formatting
        run: black --check unified4d/
      - name: Lint
        run: flake8 unified4d/
```

**문서 인프라:**
- `docs/` 폴더 → GitHub Pages 또는 Read the Docs 연동
- API 레퍼런스 자동 생성 (`sphinx` 또는 `mkdocs`)
- 변경 이력 `CHANGELOG.md` 유지

---

## 6. 커뮤니티 성장 전략

### 6.1 공개 전략: 타이밍과 채널

#### 첫 공개 (Phase 1 완료 시점)

```
Target: 첫 주 100+ 스타

채널 1 - Twitter/X (가장 중요)
  - 데모 GIF + 핵심 결과 이미지
  - "SAM 3 + Track4World를 합쳤더니..." 스토리텔링
  - @_akhaliq (Hugging Face), @gradio, @PyTorch 태그
  - 오전 9-11시 (UTC) 게시 최적

채널 2 - Reddit
  - r/MachineLearning: 기술적 상세 포스팅
  - r/computervision: 응용 중심 포스팅

채널 3 - Hugging Face
  - Model Hub 업로드 (체크포인트)
  - Spaces 데모 배포
  - Daily Papers 제출 (arXiv 공개 시)

채널 4 - LinkedIn
  - 한국어 + 영어 이중 포스팅
  - Store Guard 적용 사례 강조 (B2B 관심층)
```

#### 지속적 성장 (Phase 2-3)

```
월 1회:
  - Blog post (기술적 심층 분석)
  - "What we learned from X failures" 시리즈
  - 커뮤니티 기여 highlight

분기 1회:
  - 새 기능 릴리즈 + 성능 비교 업데이트
  - Benchmark 결과 업데이트 테이블
```

---

### 6.2 기여자 유치 전략

**Good First Issue 전략:**
```
즉시 만들 수 있는 Good First Issues:

1. "Add support for MP4 video input" (난이도: ★)
2. "Improve visualization utilities" (난이도: ★★)  
3. "Add ONNX export support" (난이도: ★★★)
4. "Benchmark on KITTI dataset" (난이도: ★★★)
5. "Add Docker support" (난이도: ★★)
```

**기여자 리텐션:**
- 첫 PR 합병 후 24시간 이내 리뷰 + 머지
- `CONTRIBUTORS.md` 유지
- 주요 기여자에게 co-author 기회 제공 (논문 기여 시)

---

### 6.3 Store Guard를 킬러 앱으로 활용

현재 Store Guard는 내부 응용 사례로만 언급되어 있습니다. 이것을 공개 데모로 활용하면 강력한 차별화 요소가 됩니다:

```
공개 가능한 Store Guard 데모:
  - CCTV 영상 (공개 데이터셋: VIRAT, UCF-Crime 등)
  - "빨간 옷 입은 사람 추적" 실시간 데모
  - occlusion 전후 identity 유지 비교
  - 3D 구역 침범 감지 시각화

주의: 실제 매장 CCTV는 프라이버시 문제로 공개 불가
     → 공개 CCTV 데이터셋 사용
```

---

## 7. 경쟁 프로젝트 분석 및 차별화

### 7.1 직접 경쟁자 분석

| 프로젝트 | 강점 | 약점 | Unified4D와의 차별화 |
|---|---|---|---|
| SAM 3 (Meta) | 텍스트 segmentation SOTA | 3D 정보 없음, occlusion 취약 | Motion prior 추가 |
| Track4World (Tencent) | 4D tracking SOTA | 텍스트 쿼리 불가 | 언어 기반 쿼리 추가 |
| Grounded SAM 2 | SAM2 + GDINO | 비디오 tracking 취약 | 4D flow integration |
| DEVA | 반자동 비디오 segmentation | 텍스트 입력 제한적 | Open-vocabulary |
| Cutie | 온라인 비디오 오브젝트 세그멘테이션 | 초기 마스크 필요 | Text-based initialization |

### 7.2 Unified4D의 진짜 차별화 포인트

경쟁 프로젝트들에 없는 것:

```
1. Motion coherence를 segmentation prior로 활용
   → "같이 움직이는 것은 같은 객체다" 원칙의 명시적 구현

2. Text + Motion dual-signal presence detection
   → "이 텍스트 개념이 존재하는가?" + "관련 motion이 있는가?"의 AND 조건

3. 3D bounding box (depth-grounded)
   → 2D bbox 대비 구역 판단의 정확도 향상 (Store Guard 핵심)

4. Flow-aware identity preservation
   → occlusion 후 재등장 시 동일 identity 유지
```

---

### 7.3 논문이 기여할 수 있는 포인트

현재 설계 기준으로 학술적으로 novel한 기여:

1. **Gated Triple Cross-Attention**: visual × text × flow의 동적 밸런싱
   - 기존: visual × text (SAM 3)
   - 기존: visual × flow (Track4World 내부)
   - 신규: 셋을 동시에 + 상황 기반 gate

2. **Flow-Aware Memory Gating**: trajectory 불연속 memory downweight
   - 기존 SAM 2 memory는 외형 기반
   - 신규: 3D motion trajectory 일관성으로 memory 신뢰도 결정

3. **Motion Coherence Map을 Segmentation Prior로 활용**
   - 학술적으로 직관적이고 검증 가능한 가설

---

## 8. 성공 지표 및 KPI

### 8.1 단기 지표 (3개월)

| 지표 | 목표 | 측정 방법 |
|---|---|---|
| GitHub Stars | 300+ | GitHub Insights |
| Colab 노트북 실행 수 | 500+ | Google Analytics |
| Issue 활동 | 20+ 이슈 오픈 | GitHub Issues |
| 첫 외부 기여자 PR | 3+ | GitHub PRs |
| Twitter 노출 | 50K+ impressions | Twitter Analytics |
| Hugging Face downloads | 1K+ | HF Hub Stats |

### 8.2 중기 지표 (6개월)

| 지표 | 목표 |
|---|---|
| GitHub Stars | 1,000+ |
| arXiv 인용 또는 블로그 언급 | 10+ |
| 논문 제출 | 1편 |
| Level 2 모델 공개 | 완료 |
| 외부 기여자 수 | 10+ |
| Discord/Slack 커뮤니티 멤버 | 200+ |

### 8.3 장기 지표 (12개월)

| 지표 | 목표 |
|---|---|
| GitHub Stars | 3,000+ |
| Level 3 Full Fusion 공개 | 완료 |
| 논문 게재 (CVPR/ICCV/NeurIPS) | 1편 |
| Store Guard 상용화 또는 기술 이전 | 1건 |
| Paperclip AI 내 Unified4D 에이전트 | 운영 중 |

---

## 9. 즉시 실행 액션 플랜 (이번 주)

### 이번 주 월요일까지

- [ ] `use_memory` 버그 수정 및 테스트
- [ ] `_SimplePixelDecoder` 4단계 upsampling으로 교체
- [ ] `tests/test_models.py` 최소 3개 테스트 작성
- [ ] embed_dim=256, depth=6 소형 모델로 smoke test 통과 확인

### 이번 주 금요일까지

- [ ] Level 1 Late Fusion 클래스 구현 완료
- [ ] DAVIS-mini (10개 시퀀스)로 SAM 3 vs Unified4D Level 1 비교 실험
- [ ] 결과 시각화 GIF 1개 이상 생성
- [ ] README.md 전면 재작성

### 다음 주까지

- [ ] Colab 노트북 초안 (15분 실행 가능)
- [ ] Hugging Face Hub에 Level 1 체크포인트 업로드
- [ ] Twitter 첫 공개 포스팅

---

## 결론: 솔직한 평가

### 지금 프로젝트의 상태

```
설계 완성도:  ████████░░  80%
코드 완성도:  ██████░░░░  60%
버그 수:      ████████░░  6개 이상 확인
실행 가능성:  ███░░░░░░░  30% (smoke test만 가능)
GitHub 준비도: ██░░░░░░░░  20%
```

### GitHub ★1K+를 위한 가장 중요한 진실

**코드의 완성도가 스타 수를 결정하지 않습니다.**

SAM (2023)은 처음 공개 당시 inference-only 코드만 있었습니다.
Stable Diffusion도 처음엔 간단한 demo script만 있었습니다.
Track Anything도 SAM + XMem을 붙인 간단한 wrapper였습니다.

**"이것이 작동하고, 기존보다 더 낫다"는 것을 30초 안에 보여줄 수 있는 GIF 하나가 900줄의 코드보다 더 많은 스타를 가져옵니다.**

Level 3 Full Fusion이 완성될 때까지 기다리지 마세요.
Level 1이 완성되는 즉시 공개하고, 커뮤니티의 반응을 보면서 Level 2, 3을 만들어 가는 것이 올바른 순서입니다.

---

*이 문서는 2026-03-19 기준으로 작성되었습니다.*  
*프로젝트 진행에 따라 분기마다 업데이트를 권장합니다.*
