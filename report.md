---
title: Homework 1 -- Multiple Object Video Detection
author: Anastasiia Yosypiv, Jiya Lal
course: Artificial Intelligence · UCU
date: 2026-10-26
---

## Introduction

1. Data understanding and preprocessing
Наш датасет містить 8 класів об'єктів: person, bicycle, car, van, truck, tricycle, bus і motor. Під час аналізу ми побачили, що більшість об'єктів мають середній розмір (63,5%), а близько чверті (24,9%) мають малий розмір. Також на одному кадрі в середньому є 45 об'єктів, а на окремих кадрах їхня кількість до 180. Оскільки дані взяті з відео, сусідні кадри часто дуже схожі, тому спираючись на медіанний IoU і на те, що ми хотіли зекономити час та GPU ми залишали кожен четвертий кадр (step = 4). Анотації COCO ми конвертували у формат YOLO, а ignored regions не враховували під час оцінювання та попередньої обробки зображень. Для валідації використовували GroupKFold погрупованим за відео, щоб кадри з одного відео не потрапляли одночасно в train і validation.

## Method

Inline math works with single dollars: $f(x) = \sigma(Wx + b)$. Display math uses double dollars:

$$
\mathcal{L}(\theta) = -\frac{1}{N}\sum_{i=1}^{N} y_i \log \hat{y}_i
$$

Code blocks are highlighted:

```python
def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()
```

## Results

| Model    | Accuracy | F1   |
|----------|----------|------|
| Baseline | 0.71     | 0.68 |
| Ours     | 0.84     | 0.82 |

Images go into the `images/` folder:

<!-- ![Training loss](images/loss.png) -->

## Conclusion

> Summarize your findings here.
