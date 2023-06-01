stable diffusion 모델을 이용한 pose synthesis 연구를 하고 싶은데, 어디서부터 시작해야 하나 보다가 읽게된 논문이다.   
이렇게 정리를 하면서 논문을 읽지 않았는데, 기왕에 읽는 논문을 정리해보자, 싶어서 시작하게 되었다 *^^*   

Microsoft 에서 2020년도에 발표된 논문인데, 당시에는 Stable Diffusion 이 많이 쓰이지 않았지만   
Pose Synthesis 를 위한 전략에 대해서 흥미있게 읽을 수 있었다.

# 1. introductios
segmentation map 이나 edge map, pose keypoint 와 같은 조건과 exemple 에서의 style 을 융합한 새로운 이미지 합성 기술을 제안한다.   
![image](https://github.com/dreamyou070/PaperReview/assets/68425947/23e61d52-0816-450a-abf3-dc3293194ff0)   
image classificatino task 에서 사전 학습된 VGG 와 같은 모델을 사용하여 스타일 이미지와 조건 이미지의 공통 domain 을 이용하는 방법을 취한다. (마치 CLIP 이 text, image 의 공통 공간을 이용하는 것처럼) 그런데 이런 사전 학습된 모델은 mask 와 같은 것을 표현하는 어려움을 기존 연구에서는 예제 이미지를 semantic region 으로 따로 분리하고, 다른 부분을 합성하는 것을 학습한다. 하지만 이러한 방법들은 범용적이지 못하다는 한계가 있다. cross-domain 이미지를 위한 dense semantic 상응을 학습하고자 하며, 이를 이용해서 이미지 전이에 활용하고자 한다. 이는 약한 지도 학습이다. 왜냐하면 우리는 correspondence annotation 도 없고, ground truth 도 없기 때문에 사실상 약한 지도학습이라는 것에 강조를 둔다. 네트워크는 크게 2개로 구성된다.
1) Cross- domain correspondence Network : 서로 다른 domain 의 이미지를 중간 domain 에 두어서, 각 이미지에 어울리는 dense 를 만들 수 있다.
2) Translation network : 공간적으로 다양한 denormalization block 을 이용해서 점진적으로 output 을 합성한다. 
<hr/>   

# 2. Approach
### 2.1 cross domaincorrespondence network
VGG network 와 같은 경우 pixel image 의 잠재 공간은 잘 형성해 주지만 semantic map 와 같은 것은 형성을 못한다. 이에, common space, S 를 형성하기 위해서 다른 전략을 사용한다.   
input image 는 F(A→S) network 에 의해서 latnet 로 표현되고, exemplar 은 F(B→S) 에 의해서 latent 로 표현된다.   
각 Feature 은 similarity 를 통해서 일치가 되게 학습되는데 이때 사용하는 방법이 **Correlation Matrix**이다.
이 matrix 는 (HW,HW) 의 형태를 가지는 2D matrix 이다.   
### 2.2 translatino network   
![image](https://github.com/dreamyou070/PaperReview/assets/68425947/105fbeae-1663-4e20-9f4e-795d20b67394)   
### 2.3 translatino network,g   
translation netork 는 근본적으로는 벡터를 image 로 변환하는 (마치 decoder 와 같은) 네트워크이다. 이를 위해 (1) spatially adaptive denormalization (SPADE Block) 을 한다.
 - 총 L 개의 layer 로 이루어져 있다. 특히나 prior 에서 구해진 structure 의 정보를 더 많이 유지하기 위해 batch normalization을 한다.
 - convolution layer 을 이용해서 constance vector 을 구하고, 여기에 z 를 이용해서 최종 이미지를 생성한다.
### 2.4 loss function   
cross domain correspondence network 와 translation network 를 end-to-end 로 학습하게 된다.
##### (1) feature matching loss   
![image](https://github.com/dreamyou070/PaperReview/assets/68425947/736ca39c-8b88-45df-afcb-c9da382c58cf)   
network 를 통과한 결과는 정답 이미지와 pixel 정보가 동일하다는 목적이다.   
##### (2) domain alignment loss   
![image](https://github.com/dreamyou070/PaperReview/assets/68425947/f7bb9892-dcd3-41f0-9863-e53833b4e0ff)
x 와 y 의 domain 을 변경한 후의 latent 가 같도록 한다.   
#### (3) exemplar translation loss   
  [1] perceptual loss : 변형 이후의 이미지와 정답 이미지의 semantic 이 같다는 목적으로, VGG 에서 deep 한 feature 만을 사용한다.   
  [2] contextual loss : 변경하기 전과 후는 전체적인 스타일이 같아야 한다는 목적으로 만들어지며, VGG 에서 shallow 한 feature 만을 사용한다.   
#### (4) Correspondence regularization   
변형을 하기 전의 vector 는 정답과 같아야 한다는 목적에서 만들어진 것이다.
#### (5) Adversarial loss   
GAN 의 구조를 이용하는 만큼 기본 loss 를 사용한다.
   
# 3. Experiments   
학습에 사용한 dataset 은   
(1) ADE20k   
(2) ADE20k-outdoor   
(3) CelebA-HQ : face image   
(4) Deepfashion : 몸 전체 이미지   
이다.

# conclusion   
이미지 연구를 시작한지 얼마 되지 않지만, 이렇게 다양한 image edit 이 가능한 것은 매우 고무적이라 보인다.   
그러나 논문에서 보여주는 결과가 전부는 아닐 수 있다.   
그래서, 실제로 해보고, 또한 이 방법은 GAN  에서 적용된 방법이므로,   
이를 Stable Diffusion 에서 구현하는 방법을 생각하는 것은 매우 바람직해 보인다.













