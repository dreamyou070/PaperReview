stable diffusion 모델을 이용한 pose synthesis 연구를 하고 싶은데, 어디서부터 시작해야 하나 보다가 읽게된 논문이다.
이렇게 정리를 하면서 논문을 읽지 않았는데, 기왕에 읽는 논문을 정리해보자, 싶어서 시작하게 되었*다 *^^*
Microsoft 에서 2020년도에 발표된 논문인데, 당시에는 Stable Diffusion 이 많이 쓰이지 않았지만 Pose Synthesis 를 위한 전략에 대해서 흥미있게 읽을 수 있었다.

1. introduction
Conditional image synthesis aims to generate photorealistic
images based on certain input data [18, 45, 52, 6].
We are interested in a specific form of conditional image
synthesis, which converts a semantic segmentation mask,
an edge map, and pose keypoints to a photo-realistic image,
given an exemplar image, as shown in Figure 1. We refer
to this form as exemplar-based image translation. It allows
more flexible control for multi-modal generation according
to a user-given exemplar.
Recent methods directly learn the mapping from a semantic
segmentation mask to an exemplar image using neural
networks [17, 38, 34, 44]. Most of these methods encode
the style of the exemplar into a latent style vector, from
which the network synthesizes images with the desired style
similar to the examplar. However, the style code only characterizes
the global style of the exemplar, regardless of spa-