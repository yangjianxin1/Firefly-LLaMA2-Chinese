# Firefly-LLaMA2-Chinese: 中文LLaMA-2大模型

<div align="left">

![GitHub Repo stars](https://img.shields.io/github/stars/yangjianxin1/Firefly-LLaMA2-Chinese?style=social)
[![Generic badge](https://img.shields.io/badge/🤗-Huggingface%20Repo-green.svg)](https://huggingface.co/YeungNLP)

</div>

<img src="pics/firefly_logo.png" width="250">

7月18日，Meta发布了LLaMA-2🦙系列模型，一夜之间，大模型的格局发生巨变。此次Meta一共发布了7B、13B、70B三种参数规模的预训练模型，
并且刷新了各个大模型榜单上的最佳成绩，一举超越此前的LLaMA-1和Falcon模型。更重要的是LLaMA-2模型开源可商用。

## 项目目标
此前[Firefly](https://github.com/yangjianxin1/Firefly)项目专注于使用有限的资源微调大模型，获得了很多开发者的关注和支持。以此为契机，我们将开展中文增量预训练的工作。

本项目的目标：
- **为广大开发者提供低成本的增量预训练方案**。授人以鱼🐟，不如授人以渔🎣，开发者可以使用本项目进行垂直领域大模型的增量预训练，提升模型的专业性，并且降低训练成本。
- **对LLaMA-2模型进行汉化**。LLaMA在2万亿token中学习到了丰富的语言知识，能有效地迁移到中文领域。我们将使用大规模中文语料，对LLaMA-2进行中文增量预训练。
- **兼容中文大模型增量预训练**。本项目将同时兼容对Baichuan、InternLM等中文大模型进行增量预训练。
- **垂直域模型**。我们会在各个垂直领域，测试本项目的实际效果。

敬请期待，欢迎大家宝贵的建议和意见。

## 技术交流
欢迎加入Firefly大模型技术交流群

<img src="pics/wechat.png" width="250">