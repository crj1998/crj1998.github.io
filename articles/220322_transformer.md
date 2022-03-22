---
id: 220322_transformer
date: 2022-03-22
title: Transform
author: crj1998@sjtu.edu.cn
---

# 从零开始的Transform
> Transformer模型是2017年Google在[Attention is All You Need](https://arxiv.org/abs/1706.03762)论文中提出的。它最大的特点是引入了注意力机制(*Attention*)，并行化的处理提升了传统RNN循环输入的速度。

## A High-Level Look at Transform
论文[Attention is All You Need](https://arxiv.org/abs/1706.03762)使用下图展示了Transformer模型结构，这幅架构图从底层细节上展示了模型，如果没有相关知识会难以理解。下面我们会从顶向下的视角解析Transformer。
<div style="text-align: center;"><img src="220322_transformer/transformer_architecture.jpg" style="width:50%;" alt="Transform architecture"/></div>
如果把Transformer模型看作一个黑盒(black box)，那么对于机器翻译(machine translation)任务而言，输入是源语言(source)中的一个句子，输出则是目标语言(target)中对应的翻译结果。
![Transform high level](220322_transformer/transformer.png =100%x*)
打开黑盒，可以看到Transformer模型由编码组件(encoders)，解码组件(decoders)和两者之间的连接组成。
<div style="text-align: center;"><img src="220322_transformer/transformer_encoders_decoders.png" style="width:60%;" alt="Transform encoders decoders"/></div>
编码组件都是由**N**个编码器堆叠串联而成，解码组件同样是由**N**个解码器堆叠。需要注意的是两个组件之间的连接，在[Attention is All You Need](https://arxiv.org/abs/1706.03762)原文中，是把解码组件的最终输出作为输入送入到每个解码组件中的解码器中。但是这样的连接方式不是唯一的方式，也可以有不同的连接方式。
<div style="text-align: center;"><img src="220322_transformer/transformer_encoder_decoder_stack.png" style="width:60%;" alt="Transform encoder decoder stack"/></div>
注意，编码组件中的**N**个编码器结构相同，但参数不同。解码组件中也是一样。每个编码器由自注意力层和前向层串联而成，自注意力层能够发现句子中单词与其他单词之间的关系，前向层则可以进行特征变换。解码器的结构与编码器相似，但是在自注意力层和前向层中加入了编码器-解码器注意力层， 这一层用于融合源域与目标域的信息。
<div style="text-align: center;"><img src="220322_transformer/transformer_encoder_decoder.png" style="width:70%;" alt="Transform encoder decoder"/></div>

参数矩阵$$W^q$$ $$W^k$$ $$W^v$$

```latex
x=\frac{ -b\pm\sqrt{ b^2-4ac } } {2a}
```



## Self-Attention

-------
## Reference
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)