from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score,recall_score

from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import numpy as np

dims = 0
classes = 0
kernel = 'rbf'


def leitura(fn):
    global dims, classes
    x, y = [], []
    with open('./dataset/'+fn+'.csv') as file_:
        for lin in file_:
            if str(lin).startswith('dims:'):
                dims = int(lin.strip('dims:'))
            elif str(lin).startswith('classes:'):
                classes = 2 if int(lin.strip('classes:')) ==1 else int(lin.strip('classes:'))
            else:
                lin = lin.strip('\n').split(',')
                x.append(list(map(float, lin[:dims])))
                if classes > 1:
                    v = np.array(list(map(float, lin[dims:])))
                    v = np.argmax(v)
                    y.append(v)
                else:
                    y.append(int(lin[-1]))
        return np.array(x), np.array(y)


def estatisticas(y, predict):
    for cl in range(classes):
        vp,fp,vn,fn =0,0,0,0
        for i in range(len(y)):
            if predict[i] == cl:
                if y[i] == predict[i]:
                    vp += 1
                else:
                    fp += 1
            else:
                if y[i] == predict[i]:
                    vn += 1
                else:
                    fn += 1
        try:
            precisao = vp / (vp + fp)
        except ZeroDivisionError:
            precisao = 0.0
        try:
            revocacao = vp / (vp + fn)
        except ZeroDivisionError:
            revocacao = 0.0
        print('\tprecisao: ', precisao)
        print('\trevocacao: ', revocacao)
        print('\tacuracia: ', (vp + vn)/(len(y)))


def svm():
    for kernel in ['linear', 'rbf']:
        print('kernel: ', kernel)
        for fn in ['MDML', 'NL', 'XOR', 'LS']:
            print('dataset: ', fn)
            x_train, y_train = leitura(fn+'_train')
            x_test, y_test = leitura(fn+'_test')
            for c in [0.1, 1, 10]:
                print('c: ', c)
                s = SVC(C=c, kernel=kernel)
                s.fit(x_train, y_train,)
                predict = s.predict(x_test)
                estatisticas(y_test, predict)
                if classes == 2:
                    plot_decision_regions(x_train, y_train, clf=s, legend=2)
                    titulo = 'SVN-'+kernel+' c='+str(c)+', dataset: '+fn+'_train'
                    plt.title(titulo)
                    plt.savefig('./relatorio/svm/'+titulo+'.png')
                    plt.show()


def knn():
    for fn in ['MDML', 'NL', 'XOR', 'LS']:
        print('dataset: ', fn)
        x_train, y_train = leitura(fn+'_train')
        x_test, y_test = leitura(fn+'_test')
        for n in [1, 2, 5, 10, 100]:
            print('k: ', n)
            k = KNeighborsClassifier(n_neighbors=n)
            k.fit(x_train, y_train,)
            predict = k.predict(x_test)
            estatisticas(y_test, predict)
            if classes == 2:
                plot_decision_regions(x_train, y_train, clf=k, legend=2)
                titulo = 'KNN - k = '+str(n)+' - '+fn+'_train'
                plt.title(titulo)
                plt.savefig('./relatorio/knn/'+titulo+'.png')
                plt.show()


if __name__ == "__main__":
    svm()
    knn()
