from curses import raw
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from sklearn.base import is_classifier, is_regressor
from sklearn.linear_model import LinearRegression

from acslibrary.methods.metrics import rmsle, r2_adjusted
from acslibrary.methods.metrics import r2_adjusted
from acslibrary.methods.metrics import rmse


def std_getfeatimp(estimator):

    e = estimator.best_estimator_ if isinstance(estimator, RandomizedSearchCV) else estimator
    try:
        imps = e.feature_importances_
    except:
        try:
            imps = e.coef_
        except:
            imps = None
    return imps


def yield_cv(train_idxs, test_idxs):
    for i in range(len(train_idxs)):
        yield train_idxs[i], test_idxs[i]


def binary_classification(
    df_Xtrain,
    df_Ytrain,
    df_Xtest=None,
    df_Ytest=None,
    parms=None,
    est=RandomForestClassifier,
    n_iter=200,
    cv=4,
    gridsearch=False,
    plot_feat_imp=True,
    num_plot_feat_imp=25,
    plot_roc=True,
    plot_cm=True,
    scorer_func=accuracy_score,
    scorer_kw={},
    scorer_greater_is_better=True,
    postprocY_func=None,
    postprocY_kw={},
    getfeatimp_func=std_getfeatimp,
):

    if not is_classifier(est):
        raise TypeError(
            "Estimator is not a classifier. If you are trying to run a regression algorithm, call method 'model_selection.regression'"
        )
    else:
        diag_kw = {
            'plot_roc': plot_roc,
            'plot_cm': plot_cm,
        }

        featimp, details, summary, reg = generate_models(
            df_Xtrain,
            df_Ytrain,
            df_Xtest,
            df_Ytest,
            parms,
            est,
            n_iter,
            cv,
            gridsearch,
            plot_feat_imp,
            num_plot_feat_imp,
            scorer_func,
            scorer_kw,
            scorer_greater_is_better,
            postprocY_func,
            postprocY_kw,
            getfeatimp_func,
            diagnostics_func=binary_classification_diagnostics,
            diagnostics_kw=diag_kw,
        )

        return featimp, details, summary, reg


def vary_params(
    X_train,
    X_test,
    y_train,
    y_test,
    parms,
    to_vary,
    values,
    n_repeats=30,
    est=RandomForestClassifier,
    scorer_func=accuracy_score,
    scorer_kw={},
):

    results = pd.DataFrame()

    for val in values:
        parms[to_vary] = val

        scores_test = []
        for i in range(n_repeats):

            model = est(**parms, random_state=i, n_jobs=-1)
            model_fit = model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = scorer_func(y_test, preds, **scorer_kw)

            scores_test += [score]

        print(
            '[Experiment {}={}] Average Test Score: {}'.format(to_vary, val, np.mean(scores_test))
        )
        results[str(val)] = scores_test

    results.boxplot()


def regression(
    df_Xtrain,
    df_Ytrain,
    df_Xtest=None,
    df_Ytest=None,
    parms=None,
    est=LinearRegression,
    n_iter=200,
    cv=4,
    gridsearch=False,
    plot_feat_imp=True,
    num_plot_feat_imp=25,
    plot_hist=True,
    plot_qq=True,
    plot_sorted_resids=False,
    sort_vec=None,
    scorer_func=mean_absolute_error,
    scorer_kw={},
    scorer_greater_is_better=False,
    postprocY_func=None,
    postprocY_kw={},
    getfeatimp_func=std_getfeatimp,
):

    if not is_regressor(est):
        raise TypeError(
            "Estimator is not a regression. If you are trying to run a classification algorithm, call method 'model_selection.binary_classification'"
        )
    else:
        if plot_sorted_resids:
            if sort_vec is None:
                raise Warning("Sorting vector is empty. Sorted residuals won't be plotted.")
        elif sort_vec is not None:
            raise Warning('Sorting vector is not empty. Sorted residuals will be plotted.')

        diag_kw = {
            'plot_hist': plot_hist,
            'plot_qq': plot_qq,
            'plot_sorted_resids': plot_sorted_resids,
            'sort_vec': sort_vec,
        }

        featimp, details, summary, reg = generate_models(
            df_Xtrain,
            df_Ytrain,
            df_Xtest,
            df_Ytest,
            parms,
            est,
            n_iter,
            cv,
            gridsearch,
            plot_feat_imp,
            num_plot_feat_imp,
            scorer_func,
            scorer_kw,
            scorer_greater_is_better,
            postprocY_func,
            postprocY_kw,
            getfeatimp_func,
            diagnostics_func=regression_diagnostics,
            diagnostics_kw=diag_kw,
        )

        return featimp, details, summary, reg


def generate_models(
    df_Xtrain,
    df_Ytrain,
    df_Xtest=None,
    df_Ytest=None,
    parms=None,
    est=None,
    n_iter=200,
    cv=4,
    gridsearch=False,
    plot_feat_imp=True,
    num_plot_feat_imp=25,
    scorer_func=None,
    scorer_kw={},
    scorer_greater_is_better=None,
    postprocY_func=None,
    postprocY_kw={},
    getfeatimp_func=std_getfeatimp,
    diagnostics_func=None,
    diagnostics_kw={},
):

    _rs = False
    is_clf = True if is_classifier(est) else False
    test_data = True if df_Xtest is not None else False
    target = df_Ytrain.name

    scorer = make_scorer(scorer_func, greater_is_better=scorer_greater_is_better, **scorer_kw)
    train_idxs, test_idxs = [], []

    if cv is not None and type(cv) != int:
        for fold in cv:
            train_idxs += [list(fold[0])]
            test_idxs += [list(fold[1])]

    if parms is not None:
        if all(type(value) == list for value in parms.values()):
            parm_list = list(ParameterGrid(parms))
        else:
            parm_list = [parms]
            gridsearch = True  # set to true so that RandomSearchCV doesn't break
        print('Grid size = ' + str(len(parm_list)))

        if gridsearch != True:
            _rs = True
            # if n_iter >= size of full grid, make gridsearchcv?
            cross_val = cv if type(cv) == int else yield_cv(train_idxs, test_idxs)
            e = RandomizedSearchCV(
                estimator=est,
                param_distributions=parms,
                n_iter=n_iter,
                cv=cross_val,
                scoring=scorer,
                verbose=0,
                random_state=42,
                n_jobs=-1,
            )
            reg = e.fit(df_Xtrain, df_Ytrain)
            parm_list = [reg.best_params_]
    else:
        parm_list = [est().get_params()]

    loop_num = 0
    appended_data_detail, appended_data_tout, appended_data_feat_imps, models = [], [], [], []
    scores = pd.DataFrame(columns=['test', 'train'])

    for p in parm_list:

        if gridsearch:
            print('')
            print('Estimator #' + str(loop_num + 1) + ' -> ' + str(p))

        if not isinstance(est, Pipeline):
            e = est(**p)
            reg = e.fit(df_Xtrain, df_Ytrain)
        raw_feat_imps = getfeatimp_func(reg)

        # Model scores (train and test)
        y_fit = reg.predict(df_Xtrain)
        y_fit_pb = reg.predict_proba(df_Xtrain)[:, 1] if is_clf else None

        if cv is not None:
            gen_cv = yield_cv(train_idxs, test_idxs) if type(cv) != int else cv
            scores_cv = cross_val_score(
                estimator=e, X=df_Xtrain, y=df_Ytrain, scoring=scorer, cv=gen_cv
            )
            score_train = round(scores_cv.mean(), 3)
            sd = ' (' + str(round(scores_cv.std(), 3)) + ')'
            score_train = score_train if scorer_greater_is_better else -score_train
            msg = 'Mean train score over CV folds = '
        else:
            score_train = scorer_func(df_Ytrain, y_fit, **scorer_kw)
            sd = ''
            msg = 'Train score = '
        print(msg + str(score_train) + sd)

        if test_data == True:
            y_pred = reg.predict(df_Xtest)
            y_pred_pb = reg.predict_proba(df_Xtest)[:, 1] if is_clf else None
            score_test = scorer_func(df_Ytest, y_pred, **scorer_kw)
            print('Test score = ' + str(score_test))

            if postprocY_func is not None:
                y_test, y_pred, y_pred_pb = postprocY_func(
                    df_Ytest, y_pred, y_pred_pb, **postprocY_kw
                )
            else:
                y_test, y_pred, y_pred_pb = df_Ytest, y_pred, y_pred_pb
        else:
            score_test = np.nan

        if postprocY_func is not None:
            y_train, y_fit, y_fit_pb = postprocY_func(df_Ytrain, y_fit, y_fit_pb, **postprocY_kw)
        else:
            y_train, y_fit, y_fit_pb = df_Ytrain, y_fit, y_fit_pb

        scores = scores.append({'test': score_test, 'train': score_train}, ignore_index=True)

        # Create feat_imps dataset
        if raw_feat_imps is not None:
            feat_names = df_Xtrain.columns.values
            if len(raw_feat_imps) != len(feat_names):
                feat_names = np.array([f'x{i}' for i in range(len(raw_feat_imps))])
            feat_imps = pd.DataFrame(zip(feat_names, raw_feat_imps))
            feat_imps.columns = ['column', 'importance']
            feat_imps['score_test'] = score_test
            feat_imps['score_train'] = score_train
            feat_imps['random_search'] = _rs
            feat_imps['est'] = str(p)
            feat_imps['ID'] = loop_num

        # Create detailed dataset (all observations)
        if is_clf:
            tout = pd.DataFrame(zip(y_fit, y_fit_pb), columns=['pred', 'prob']).set_index(
                y_train.index
            )
        else:
            tout = pd.DataFrame(y_fit, columns=['pred']).set_index(y_train.index)

        tout = pd.DataFrame(y_train).join(tout)
        tout['test'] = False

        if test_data:

            if is_clf:
                tout_test = pd.DataFrame(
                    zip(y_pred, y_pred_pb), columns=['pred', 'prob']
                ).set_index(y_test.index)
            else:
                tout_test = pd.DataFrame(y_pred, columns=['pred']).set_index(y_test.index)

            tout_test = pd.DataFrame(y_test).join(tout_test)
            tout_test['test'] = True
            tout = pd.concat([tout_test, tout])

        if is_clf:
            tout.loc[(tout['pred'] >= 1.0) & (tout[target] >= 1.0), 'tp'] = 1
            tout.loc[tout['tp'].isnull(), 'tp'] = 0
            tout.loc[(tout['pred'] >= 1.0) & (tout[target] < 1.0), 'fp'] = 1
            tout.loc[tout['fp'].isnull(), 'fp'] = 0
            tout.loc[(tout['pred'] < 1.0) & (tout[target] < 1.0), 'tn'] = 1
            tout.loc[tout['tn'].isnull(), 'tn'] = 0
            tout.loc[(tout['pred'] < 1.0) & (tout[target] >= 1.0), 'fn'] = 1
            tout.loc[tout['fn'].isnull(), 'fn'] = 0

        tout['score_test'] = score_test
        tout['score_train'] = score_train
        tout['random_search'] = _rs
        tout['est'] = str(p)
        tout['ID'] = loop_num

        # Create summary dataset (based on tout above)

        f4 = tout.head(1)[['score_test', 'score_train', 'random_search', 'est']].reset_index(
            drop=True
        )
        f4['ID'] = loop_num

        def summarize(sfx, y_true, preds):
            if is_clf:
                sub_tout = tout[tout.test] if sfx == 'test' else tout[~tout.test]
                tout_sum = sub_tout.sum()
                f4['target_' + sfx] = tout_sum.loc[target]
                f4['pred_' + sfx] = tout_sum.loc['pred']
                f4['tp_' + sfx] = tout_sum.loc['tp']
                f4['fp_' + sfx] = tout_sum.loc['fp']
                f4['tn_' + sfx] = tout_sum.loc['tn']
                f4['fn_' + sfx] = tout_sum.loc['fn']
                f4['fpr_' + sfx] = f4['fp_' + sfx] / (f4['fp_' + sfx] + f4['tn_' + sfx])
                f4['tpr_' + sfx] = f4['tp_' + sfx] / (f4['tp_' + sfx] + f4['fn_' + sfx])
                f4['tp_fp_' + sfx] = f4['tp_' + sfx] / f4['fp_' + sfx]
                f4['accuracy_' + sfx] = accuracy_score(y_true, preds)
                f4['precision_' + sfx] = precision_score(y_true, preds)
                f4['f1_' + sfx] = f1_score(y_true, preds)
                f4['recall_' + sfx] = recall_score(y_true, preds)
                f4['roc_auc_' + sfx] = roc_auc_score(y_true, preds)
            else:
                x_dim = df_Xtrain.shape[1]
                f4['rmse_' + sfx] = rmse(y_true.values, preds)
                f4['mae_' + sfx] = mean_absolute_error(y_true.values, preds, **scorer_kw)
                f4['r2_' + sfx] = r2_score(y_true.values, preds)
                f4['rmsle_' + sfx] = rmsle(y_true.values, preds)
                f4['r2_adjusted_' + sfx] = r2_adjusted(y_true.values, preds, x_dim)

        summarize('train', y_train, y_fit)
        if test_data:
            summarize('test', y_test, y_pred)

        # Append to master files
        appended_data_tout.append(f4.set_index(['ID', 'est']))
        appended_data_detail.append(tout.set_index(['ID', 'est'], append=True))

        if raw_feat_imps is not None:
            appended_data_feat_imps.append(feat_imps.set_index(['ID', 'est']))

        if test_data:
            models += [
                {'estimator': reg, 'y_true': y_test, 'y_pred': y_pred, 'y_pred_pb': y_pred_pb}
            ]
        else:
            models += [
                {'estimator': reg, 'y_true': y_train, 'y_pred': y_fit, 'y_pred_pb': y_fit_pb}
            ]

        loop_num += 1

    # Create the final concatenated/appended datasets
    df_summary = pd.concat(appended_data_tout)
    df_detailed = pd.concat(appended_data_detail)

    if len(appended_data_feat_imps) != 0:
        df_feat_imps = pd.concat(appended_data_feat_imps)
    else:
        df_feat_imps = None

    # Find best model
    asc = False if scorer_greater_is_better else True
    best = scores.sort_values(['test', 'train'], ascending=asc).first_valid_index()
    best_score = scores.test[best] if test_data else scores.train[best]
    reg = models[best]['estimator']
    y_true = models[best]['y_true']
    y_pred = models[best]['y_pred']
    y_pred_pb = models[best]['y_pred_pb']

    if isinstance(reg, RandomizedSearchCV):
        bp = str(reg.best_params_)
    else:
        bp = str(reg.get_params())

    print('')
    print('Best model: ' + bp)
    print('')
    print('Best score: ' + str(best_score))

    if plot_feat_imp == True:

        raw_feat_imps = getfeatimp_func(reg)
        if raw_feat_imps is not None:

            imp_idx = np.argsort(raw_feat_imps)
            tree_indices = np.arange(0, len(raw_feat_imps)) + 0.5
            npfi = int(-1.0 * num_plot_feat_imp)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            ax1.barh(tree_indices[:num_plot_feat_imp], raw_feat_imps[imp_idx[npfi:]], height=0.7)
            ax1.set_yticklabels(feat_names[imp_idx[npfi:]])
            ax1.set_yticks(tree_indices[:num_plot_feat_imp])
            ax1.set_ylim((0, len(raw_feat_imps[npfi:])))

            if hasattr(reg, 'estimators_'):
                filist = pd.DataFrame([list(getfeatimp_func(tree)) for tree in reg.estimators_])
                filist = filist[imp_idx[npfi:]]
                ax2.boxplot(
                    np.array(filist), vert=False, labels=df_Xtrain.columns.values[imp_idx[npfi:]]
                )

            fig.tight_layout()
            plt.show()

    diagnostics_func(y_true, y_pred, y_pred_pb, **diagnostics_kw)
    return df_feat_imps, df_detailed, df_summary, reg


def regression_diagnostics(
    y_true,
    y_pred,
    y_pred_pb=None,
    plot_hist=True,
    plot_qq=True,
    plot_sorted_resids=False,
    sort_vec=None,
):

    if plot_hist:
        plt.figure()
        resids = np.subtract(y_true, y_pred)
        m = np.mean(resids)
        s = np.std(resids)
        ax = sns.distplot(resids)
        ax.vlines(x=m - 1.96 * s, ymin=0, ymax=0.9, color='red')
        ax.vlines(x=m + 1.96 * s, ymin=0, ymax=0.9, color='red')
        plt.title('Histogram of Residuals')
        plt.xlabel('Residual')
        plt.ylabel('Frequency')
        plt.show()

    if plot_qq:
        plt.figure()
        resids = np.array(np.subtract(y_true, y_pred))
        ss.probplot(resids.flatten(), plot=plt)
        plt.title('Normal Q-Q Plot')
        plt.xlabel('Theoretical Quantile')
        plt.ylabel('Residual')
        plt.show()

    if plot_sorted_resids:
        resids = y_true - y_pred
        resids.name = 'resids'
        resids_sorted = (
            pd.DataFrame(resids)
            .merge(sort_vec, left_index=True, right_index=True)
            .sort_values(sort_vec.name, ascending=True)
        )
        resids_sorted['rolling_mean'] = resids_sorted.resids.rolling(40).mean()

        plt.figure(figsize=(10, 5))
        sns.scatterplot(data=resids_sorted, x=sort_vec.name, y='resids', color='grey')
        sns.lineplot(data=resids_sorted, x=sort_vec.name, y='rolling_mean', color='purple')
        plt.show()


def binary_classification_diagnostics(y_true, y_pred, y_pred_pb=None, plot_roc=False, plot_cm=True):

    # Plot ROC curve
    if plot_roc == True:
        fpr_pb, tpr_pb, _ = roc_curve(y_true, y_pred_pb)

        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_pb, tpr_pb, label='RF')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.show()

    # Plot confusion matrix
    if plot_cm == True:
        cm = confusion_matrix(y_true, y_pred)
        cm_display = ConfusionMatrixDisplay(cm).plot()
