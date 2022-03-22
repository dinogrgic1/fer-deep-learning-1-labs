from sklearn import svm

class KSVMWrap:
    '''
    Metode:
    __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        Konstruira omotač i uči RBF SVM klasifikator
        X, Y_:           podatci i točni indeksi razreda
        param_svm_c:     relativni značaj podatkovne cijene
        param_svm_gamma: širina RBF jezgre

    predict(self, X)
        Predviđa i vraća indekse razreda podataka X

    get_scores(self, X):
        Vraća klasifikacijske mjere
        (engl. classification scores) podataka X;
        ovo će vam trebati za računanje prosječne preciznosti.

    support
        Indeksi podataka koji su odabrani za potporne vektore
    '''