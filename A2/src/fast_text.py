import fasttext as ft

model = ft.cbow('train.txt', 'model', lr=0.1, dim=300)
classifier = ft.supervised('train.txt', 'model', label_prefix='__myprefix__',
                           thread=4)
