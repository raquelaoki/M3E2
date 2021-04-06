
#add bart

#testing

#deconfounder
class deconfounder_algorithm():
    def __init__(self):
        super(deconfounder_algorithm, self).__init__()

    def deconfounder_PPCA_LR(self, train, colnames, y01, k = 5,  b = 100):
        '''
        input:
        - train dataset
        - colnames or possible causes
        - y01: outcome
        - k: dimension of latent space
        - b: number of bootstrap samples
        '''
        x_train, x_val, holdout_mask = daHoldout(train,0.2)
        w,z, x_gen = fm_PPCA(x_train,k,True)
        #filename = 'dappcalr_' +str(k)+'_'+name
        pvalue= daPredCheck(x_val,x_gen,w,z, holdout_mask)
        alpha = 0.05 #for the IC test on outcome model
        low = stats.norm(0,1).ppf(alpha/2)
        up = stats.norm(0,1).ppf(1-alpha/2)
        #To speed up, I wont fit the PPCA to each boostrap iteration
        del x_gen
        if 0.1 < pvalue and pvalue < 0.9:
            print('Pass Predictive Check:', filename, '(',str(pvalue),')' )
            coef= []
            pca = np.transpose(z)
            for i in range(b):
                #print(i)
                rows = np.random.choice(train.shape[0], int(train.shape[0]*0.85), replace=False)
                X = train[rows, :]
                y01_b = y01[rows]
                pca_b = pca[rows,:]
                #w,pca, x_gen = fm_PPCA(X,k)
                #outcome model
                coef_, _ = outcome_model_ridge(X,colnames, pca_b,y01_b,False,filename)
                coef.append(coef_)


            coef = np.matrix(coef)
            coef = coef[:,0:train.shape[1]]
            #Building IC
            coef_m = np.asarray(np.mean(coef,axis=0)).reshape(-1)
            coef_var = np.asarray(np.var(coef,axis=0)).reshape(-1)
            coef_z = np.divide(coef_m,np.sqrt(coef_var/b))
            #1 if significative and 0 otherwise
            coef_z = [ 1 if c>low and c<up else 0 for c in coef_z ]


            #https://abdalimran.github.io/2019-06-01/Drawing-multiple-ROC-Curves-in-a-single-plot
            '''
            if ROC = TRUE, outcome model receive entire dataset, but internally split in training
            and testing set. The ROC results and score is just for testing set
            '''
            del X,pca,pca_b,y01_b
            del coef_var, coef, coef_
            w,z, x_gen = fm_PPCA(train,k,False)
            _,roc =  outcome_model_ridge(train,colnames, np.transpose(z),y01,True,filename)
            #df_ce =pd.merge(df_ce, causal_effect,  how='left', left_on='genes', right_on = 'genes')
            #df_roc[name_PCA]=roc
            #aux = pd.DataFrame({'model':[name_PCA],'gamma':[gamma],'gamma_l':[cil],'gamma_u':[cip]})
            #df_gamma = pd.concat([df_gamma,aux],axis=0)
            #df_gamma[name_PCA] = sparse.coo_matrix((gamma_ic),shape=(1,3)).toarray().tolist()
        else:
            coef_m = []
            coef_z = []
            roc = []

        return np.multiply(coef_m,coef_z), coef_m, roc, filename



    def fm_PPCA(train,latent_dim, flag_pred):
        '''
        Fit the PPCA
        input:
            train: dataset
            latent_dim: size of latent variables
            flag_pred: if values to calculate the predictive check should be saved

        output: w and z values, generated sample to predictive check

        '''
        #Reference: https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Probabilistic_PCA.ipynb
        from tensorflow.keras import optimizers
        import tensorflow as tf #.compat.v2
        import tensorflow_probability as tfp
        from tensorflow_probability import distributions as tfd
        tf.enable_eager_execution()

        num_datapoints, data_dim = train.shape
        x_train = tf.convert_to_tensor(np.transpose(train),dtype = tf.float32)


        Root = tfd.JointDistributionCoroutine.Root
        def probabilistic_pca(data_dim, latent_dim, num_datapoints, stddv_datapoints):
          w = yield Root(tfd.Independent(
              tfd.Normal(loc=tf.zeros([data_dim, latent_dim]),
                         scale=2.0 * tf.ones([data_dim, latent_dim]),
                         name="w"), reinterpreted_batch_ndims=2))
          z = yield Root(tfd.Independent(
              tfd.Normal(loc=tf.zeros([latent_dim, num_datapoints]),
                         scale=tf.ones([latent_dim, num_datapoints]),
                         name="z"), reinterpreted_batch_ndims=2))
          x = yield tfd.Independent(tfd.Normal(
              loc=tf.matmul(w, z),
              scale=stddv_datapoints,
              name="x"), reinterpreted_batch_ndims=2)

        #data_dim, num_datapoints = x_train.shape
        stddv_datapoints = 1

        concrete_ppca_model = functools.partial(probabilistic_pca,
            data_dim=data_dim,
            latent_dim=latent_dim,
            num_datapoints=num_datapoints,
            stddv_datapoints=stddv_datapoints)

        model = tfd.JointDistributionCoroutine(concrete_ppca_model)

        w = tf.Variable(np.ones([data_dim, latent_dim]), dtype=tf.float32)
        z = tf.Variable(np.ones([latent_dim, num_datapoints]), dtype=tf.float32)

        target_log_prob_fn = lambda w, z: model.log_prob((w, z, x_train))
        losses = tfp.math.minimize(lambda: -target_log_prob_fn(w, z),
                                   optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
                                   num_steps=200)

        qw_mean = tf.Variable(np.ones([data_dim, latent_dim]), dtype=tf.float32)
        qz_mean = tf.Variable(np.ones([latent_dim, num_datapoints]), dtype=tf.float32)
        qw_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([data_dim, latent_dim]), dtype=tf.float32))
        qz_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([latent_dim, num_datapoints]), dtype=tf.float32))
        def factored_normal_variational_model():
          qw = yield Root(tfd.Independent(tfd.Normal(
              loc=qw_mean, scale=qw_stddv, name="qw"), reinterpreted_batch_ndims=2))
          qz = yield Root(tfd.Independent(tfd.Normal(
              loc=qz_mean, scale=qz_stddv, name="qz"), reinterpreted_batch_ndims=2))

        surrogate_posterior = tfd.JointDistributionCoroutine(
            factored_normal_variational_model)

        losses = tfp.vi.fit_surrogate_posterior(
            target_log_prob_fn,
            surrogate_posterior=surrogate_posterior,
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
            num_steps=400)



        x_generated = []
        if flag_pred:
            for i in range(50):
                _, _, x_g = model.sample(value=surrogate_posterior.sample(1))
                x_generated.append(x_g.numpy()[0])

        w, z = surrogate_posterior.variables

        return w.numpy(),z.numpy(), x_generated

    def daHoldout(train,holdout_portion):
        '''
        Hold out a few values from train set to calculate predictive check
        '''
        num_datapoints, data_dim = train.shape
        n_holdout = int(holdout_portion * num_datapoints * data_dim)

        holdout_row = np.random.randint(num_datapoints, size=n_holdout)
        holdout_col = np.random.randint(data_dim, size=n_holdout)
        holdout_mask = (sparse.coo_matrix((np.ones(n_holdout), \
                                    (holdout_row, holdout_col)), \
                                    shape = train.shape)).toarray()

        holdout_subjects = np.unique(holdout_row)
        holdout_mask = np.minimum(1, holdout_mask)

        x_train = np.multiply(1-holdout_mask, train)
        x_vad = np.multiply(holdout_mask, train)
        return x_train, x_vad,holdout_mask

    def daPredCheck(x_val,x_gen,w,z,holdout_mask):
        '''
        calculate the predictive check
        input:
            x_val: observed values
            x_gen: generated values
            w, z: from fm_PPCA
            holdout_mask
        output: pvalue from the predictive check


        '''
        holdout_mask1 = np.asarray(holdout_mask).reshape(-1)
        x_val1 = np.asarray(x_val).reshape(-1)
        x1 = np.asarray(np.multiply(np.transpose(np.dot(w,z)), holdout_mask)).reshape(-1)
        del x_val
        x_val1 = x_val1[holdout_mask1==1]
        x1= x1[holdout_mask1==1]
        pvals =[]

        for i in range(len(x_gen)):
            generate = np.transpose(x_gen[i])
            holdout_sample = np.multiply(generate, holdout_mask)
            holdout_sample = np.asarray(holdout_sample).reshape(-1)
            holdout_sample = holdout_sample[holdout_mask1==1]
            x_val_current = stats.norm(holdout_sample, 1).logpdf(x_val1)
            x_gen_current = stats.norm(holdout_sample, 1).logpdf(x1)

            pvals.append(np.mean(np.array(x_val_current<x_gen_current)))


        overall_pval = np.mean(pvals)
        return overall_pval

    def outcome_model_ridge(x, colnames,x_latent,y01_b,roc_flag,name):
        '''
        outcome model from the DA
        input:
        - x: training set
        - x_latent: output from factor model
        - colnames: x colnames or possible causes
        - y01: outcome
        -name: roc name file
        '''
        import scipy.stats as st
        model = linear_model.SGDClassifier(penalty='l2', alpha=0.1, l1_ratio=0.01,loss='modified_huber', fit_intercept=True,random_state=0)
        if roc_flag:
            #use testing and training set
            x_aug = np.concatenate([x,x_latent],axis=1)
            X_train, X_test, y_train, y_test = train_test_split(x_aug, y01_b, test_size=0.33, random_state=42)
            modelcv = calibration.CalibratedClassifierCV(base_estimator=model, cv=5, method='isotonic').fit(X_train, y_train)
            coef = []

            pred = modelcv.predict(X_test)
            predp = modelcv.predict_proba(X_test)
            predp1 = [i[1] for i in predp]
            print('F1:',f1_score(y_test,pred),sum(pred),sum(y_test))
            print('Confusion Matrix', confusion_matrix(y_test,pred))
            fpr, tpr, _ = roc_curve(y_test, predp1)
            auc = roc_auc_score(y_test, predp1)
            roc = {'learners': name,
                   'fpr':fpr,
                   'tpr':tpr,
                   'auc':auc}
        else:
            #don't split dataset
            x_aug = np.concatenate([x,x_latent],axis=1)
            model.fit(x_aug, y01_b)
            coef = model.coef_[0]
            roc = {}

        return coef, roc
