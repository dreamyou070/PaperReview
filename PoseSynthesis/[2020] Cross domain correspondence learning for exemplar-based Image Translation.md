stable diffusion 모델을 이용한 pose synthesis 연구를 하고 싶은데, 어디서부터 시작해야 하나 보다가 읽게된 논문이다.   이렇게 정리를 하면서 논문을 읽지 않았는데, 기왕에 읽는 논문을 정리해보자, 싶어서 시작하게 되었다 *^^*   

Microsoft 에서 2020년도에 발표된 논문인데, 당시에는 Stable Diffusion 이 많이 쓰이지 않았지만 Pose Synthesis 를 위한 전략에 대해서 흥미있게 읽을 수 있었다.

# 1. introductios
segmentation map 이나 edge map, pose keypoint 와 같은 조건과 exemple 에서의 style 을 융합한 새로운 이미지 합성 기술을 제안한다.
![image](https://github.com/dreamyou070/PaperReview/assets/68425947/23e61d52-0816-450a-abf3-dc3293194ff0)
image classificatino task 에서 사전 학습된 VGG 와 같은 모델을 사용하여 스타일 이미지와 조건 이미지의 공통 domain 을 이용하는 방법을 취한다. (마치 CLIP 이 text, image 의 공통 공간을 이용하는 것처럼) 그런데 이런 사전 학습된 모델은 mask 와 같은 것을 표현하는 어려움을 기존 연구에서는 예제 이미지를 semantic region 으로 따로 분리하고, 다른 부분을 합성하는 것을 학습한다. 하지만 이러한 방법들은 범용적이지 못하다는 한계가 있다. cross-domain 이미지를 위한 dense semantic 상응을 학습하고자 하며, 이를 이용해서 이미지 전이에 활용하고자 한다. 이는 약한 지도 학습이다. 왜냐하면 우리는 correspondence annotation 도 없고, ground truth 도 없기 때문에 사실상 약한 지도학습이라는 것에 강조를 둔다. 네트워크는 크게 2개로 구성된다.
1) Cross- domain correspondence Network : 서로 다른 domain 의 이미지를 중간 domain 에 두어서, 각 이미지에 어울리는 dense 를 만들 수 있다.
2) Translation network : 공간적으로 다양한 denormalization block 을 이용해서 점진적으로 output 을 합성한다. 
<hr/>

# 2. Approach
### 2.1 cross domaincorrespondence network
