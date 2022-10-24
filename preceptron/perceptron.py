class VotePerceptron:
    def __init__(self):
        self.reset_model()
        
    def reset_model(self):
        self.epoch = 0
        self.epoch_error_count_list = []
        self.error_arr = None
        self.w_arr = None
        self.w = None

    def import_data(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def fit(self, max_epoch, lr0=0.99, decay_epoch=10, decay_rate=0.01, min_lr=0.01): 
        self.reset_model()
        x = self.x_train
        y = self.y_train
        self.w = np.zeros(x.shape[-1])
        lr = lr0

        self.w_arr = self.w #np.vstack((self.w_arr, self.w))
        self.error_arr = np.array(self.error_rate(self.predict(x), y))
        self.epoch_error_count_list.append(self.error_count(self.predict(x), y))
        
        while self.epoch_error_count_list[self.epoch] != 0 and self.epoch < max_epoch:
            error = 0
            self.epoch += 1
            epoch_error_count = 0
            for i in range(len(x)):
                xi,yi = x[i, :], y[i]
                if self.predict(xi) != yi:
                    error += 1
                    self.w_arr = np.vstack((self.w_arr, self.w))
                    self.error_arr = np.hstack((self.error_arr, self.error_rate(self.predict(x), y)))
                    self.w += yi*xi*lr#(lr - decay_epoch * self.epoch//10)
            if lr > min_lr:
                lr = lr*(1-self.epoch//decay_epoch*decay_rate)
            self.epoch_error_count_list.append(error)    

    def error_count(self, y_predict, y):
        return sum(~np.equal(y_predict, y))

    def error_rate(self, y_predict, y):
        return self.error_count(y_predict, y)/len(y)

    def predict(self, x):
        return np.sign(np.dot(self.w, x.T))
    
    def error_plot(self):
        plt.plot([i/len(self.x_train) for i in self.epoch_error_count_list])

    def vote_predict(self, x):
        return np.sign(np.sign(np.dot(self.w_arr, x.T)).sum(axis=0))