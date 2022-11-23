---
id: 123
date: 2022-03-25
title: Python
author: crj1998@sjtu.edu.cn
---

# Hello, world!
> a guide for markdown to html
> blockquote2

------

```
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
```
The `Encoder` is inherit form `nn.Module`.

[hello good morning](hello.html)

line: hello__world__next
lien: ~~strikethrough~~

 - [x] This task is done
 - [ ] This is still pending


| h1    |    h2   |      h3 |
|:------|:-------:|--------:|
| 100   | [a][1]  | ![b][2] |
| *foo* | **bar** | ~~baz~~ |

![1](https://upload-images.jianshu.io/upload_images/13843118-059356f7d8ad8b29.JPG?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240=50%x* )
<div style="text-align: center;"><img src="https://upload-images.jianshu.io/upload_images/13843118-4b808bae68eb861d.JPG" style="width:100px;"/></div>

------

------
- reference1
- reference2
- reference3
- reference4