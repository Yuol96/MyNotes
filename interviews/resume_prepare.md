## 自我介绍

- 现在是北大大四的学生
- 17-18学年为了转行cs，选择gap了一年，学了很多基础课，在信科网络所的一个课题组里做科研，主要是网络数据流的算法和P4数据平面编程；后来去软件所做了一些可视化的研究项目；还在18年后半年去Face++实习，主要做计算机视觉与深度学习相关的内容。上学期恢复学籍后，通过选课系统选了很多CS的基础课程，同时在申请美国的CS master项目。最近已经收到了UCSD的CS ms的offer，预计19年秋季就会去美国读书，因此希望在出国之前能在hulu积累实习经验。
- 在软件所发的那篇一作文章昨天刚刚被接收，主要是论文引用的影响力图的可视化研究和可视化系统。

## 项目经历 

(1) motivation (2) contribution (3) literature (4) model (if any) (5) methodology (6) data (7) results (8) conclusion (9) questions and discussion

### One-shot Object Detection: SiamRPN 

- motivation: 有个产品需求，给一个(未训练过的)物体的模板图片，要能够在其他的图片数据集中找到这个物体以及它的位置。e.g. 在一堆图片中寻找藏独分子的雪山狮子旗。
- literature: 
  - CVPR18中很火的SiamRPN，做SOT的。One-shot detection这个概念和SOT很接近，希望能把用于tracking的SiamRPN移植过来做detection。与SOT的区别：
    - SOT更简单，因为第一帧与后续桢的物体比较像
    - SOT由于相邻桢比较接近，回归和分类仅仅基于上一桢的结果，所以难度会小很多
  - ECCV18中CBAM轻量级的Attention模块
- model: <u>模板图片</u>和<u>待检测图片</u>形成两个分支，其深层特征相互卷积，分别得到classification和regression的输出  <span style="color:red;">需要能手画模型的示意图</span> 
- 数据: 
  - SOT的数据。由于是视频数据，量太少。
  - 合成数据：自己用爬虫爬了所有国家的国旗的图片，经过scale旋转扭曲透明度调整后，贴到100k+的背景图片上


- 项目的难点？存在的问题？
  - 缺少数据，只能自己建立训练测试和benchmark的数据集
  - 原始的SiamRPN没有源码，网上复现的开源版本很难收敛，只能自己复现
  - Siamese的结构不能直接以batch的方式训练，严重影响训练速度和效果
  - 回归输出不够准确
  - 捕捉颜色的能力比较强，但捕捉全局pattern的能力不够，容易形成局部匹配
- 再给两个月，如何优化？
  - 解决局部匹配：将模板图片分块之后分别与待检测图片卷积，再合并
  - 回归输出不准确：可能是模型能力的问题，需要优化回归分支

### Zero-shot Learning

- motivation: 综合调研，给个pre，看看有没有实际应用的可能
- literature: 
  - 分类问题定义：给一组未训练的物体的语义描述，和一张图片，通过模型给图片中的物体分类
  - 训练：物体的语义描述通过word2vec/属性值转换成语义空间的点，图片经过CNN转换成高位视觉空间的点，关键是通过zero-shot模型构建语义空间与视觉空间的映射关系

- 项目的难点？存在的问题？
  - domain shift: "has tail" 对猪尾巴和马尾巴有相同的语义，但在视觉空间中相差很大。
    SAE的结构可以部分缓解这个问题。
  - hubness：总是存在某些语义描述，映射后，不论输入的图片是什么，它都是最近的。the curse of dimensionality.
    不用待检测图片到各个语义描述类的直接距离，而改用待检测图片在一组输入图片中到某个语义描述类的距离的rank
- 再给两个月，如何优化？

### Eiffel

- motivation: 继续前人没做完的工作
- 论文引用关系图G —> (Node Summarization by SymNMF) —> k个cluster的图M —> (MWST) —> 去掉不重要的边M' —> 图展示算法展示在前端页面

### t-SNE

### DSAB

### Finding Significant Items in Data Stream

### P4 Programmable Switch



(1) motivation (2) contribution (3) literature (4) model (if any) (5) methodology (6) data (7) results (8) conclusion (9) questions and discussion