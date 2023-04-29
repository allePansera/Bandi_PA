from library.Training.Training import Training

t = Training(method="RF", sampler="OS_SVM")
t.train(read_existing=False, read_normalized=True)

