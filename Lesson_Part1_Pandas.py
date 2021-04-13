import numpy as np
import pandas as pd

# Тема “Вычисления с помощью Numpy”
# Задание 1
a = np.array([[1, 6],
              [2, 8],
              [3, 11],
              [3, 10],
              [1, 7]])
mean_a = a.mean(axis=0)
print(mean_a)

# Задание 2

a_centred = a - a.mean(axis=0)
print(a_centred)

# Задание 3

t0 = np.array(a_centred[:, 0])
t1 = np.array(a_centred[:, 1])

a_centred_sp = t0.dot(t1)
print(a_centred_sp / (t0.size - 1))

# Задание 4**

# m_edinichnaya = np.array([[1, 0, 0, 0, 0],
#                           [0, 1, 0, 0, 0],
#                           [0, 0, 1, 0, 0],
#                           [0, 0, 0, 1, 0],
#                           [0, 0, 0, 0, 1]
#                           ])
I = np.eye(2)
E = a.dot(I)
print(np.cov(a.transpose()))

# Тема “Вычисления с помощью Numpy”
# Задание 1
a = {"author_id": [1, 2, 3], "author_name": ["Тургенев", "Чехов", "Островский"]}
b = pd.DataFrame(a)
print(b)
e = {"author_id": [1, 1, 1, 2, 2, 3, 3],
     "book_title": ['Отцы и дети', 'Рудин', 'Дворянское гнездо', 'Толстый и тонкий', 'Дама с собачкой', 'Гроза',
                    'Таланты и поклонники'], "price": [450, 300, 350, 500, 450, 370, 290]}
c = pd.DataFrame(e)
print(c)

# Задание 2

authors_price = pd.merge(c, b, on="author_id", how="inner")
print(authors_price)

# Задание 3
top5 = authors_price.sort_values(by="price", ascending=False).head(5)
print(top5)

# Задание 4

agg_func_math = {
    'price': ['min', 'max', 'mean']
}

j = authors_price.groupby(['author_name']).agg(agg_func_math).round(2)
namesList = ['min_price', 'max_price', 'mean_price']
j.columns = namesList

print(j)

# Задание 5**
authors_price['cover'] = ['твердая', 'мягкая', 'мягкая', 'твердая', 'твердая', 'мягкая', 'мягкая']
print(authors_price)

g = pd.pivot_table(authors_price, index="author_name", columns="cover", values="price", aggfunc='sum')
print(g)

