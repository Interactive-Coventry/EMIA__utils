from libs.foxutils.utils.data_generators import multiple_point_window_generator, single_step_window_generator, \
    conv_window_generator, multiple_point_conv_window_generator, multi_step_window_generator
from libs.foxutils.utils.keras_models import run_single_step_model, run_multi_step_model
from libs.foxutils.utils import train_functionalities

from . import display_utils
import numpy as np
import pandas as pd
import warnings


SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 20
HISTORY_LENGTH = 3


def prepare_generators(train_df, val_df, test_df, target_column, label_length, history_length=HISTORY_LENGTH,
                       batch_size=BATCH_SIZE, val_percentage=None, test_percentage=None):
    wide_window = multiple_point_window_generator(train_df, val_df, test_df, target_column, label_length, batch_size,
                                                  val_percentage, test_percentage)
    print('\nResult window for plotting')
    print('Input shape:', wide_window.example[0].shape)

    single_step_window = single_step_window_generator(train_df, val_df, test_df, target_column, batch_size,
                                                      val_percentage, test_percentage)
    print(f'\nWindow generator\n{single_step_window} \nfor historylength = {1}\n')

    for example_inputs, example_labels in single_step_window.train.take(1):
        print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
        print(f'Labels shape (batch, time, features): {example_labels.shape}')

    conv_window_gen = conv_window_generator(train_df, val_df, test_df, target_column, history_length, batch_size,
                                            val_percentage, test_percentage)
    print(f'\nWindow generator\n{conv_window_gen} \nfor historylength = {history_length}\n')
    # conv_window_gen.plot()

    wide_conv_window = multiple_point_conv_window_generator(train_df, val_df, test_df, target_column, history_length,
                                                            batch_size, val_percentage, test_percentage)
    print('\nResult window for plotting')
    print('Input shape:', wide_conv_window.example[0].shape)

    return single_step_window, wide_window, conv_window_gen, wide_conv_window


def get_single_step_prediction_model(model_name, train_df, val_df, test_df, target_column, label_length,
                                     history_length=HISTORY_LENGTH, batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS,
                                     optimizer=None, val_percentage=None, test_percentage=None):
    single_step_window, wide_window, conv_window_gen, wide_conv_window = prepare_generators(train_df, val_df, test_df,
                                                                                            target_column, label_length,
                                                                                            history_length=history_length,
                                                                                            batch_size=batch_size,
                                                                                            val_percentage=val_percentage,
                                                                                            test_percentage=test_percentage, )
    if type(train_df) is dict:
        feature_names = train_df[list(train_df.keys())[0]].columns
    else:
        feature_names = train_df.columns
    label_index = feature_names.tolist().index(target_column)

    if model_name == 'baseline':
        model, val_performance, test_performance, _ = run_single_step_model('baseline', single_step_window,
                                                                            wide_window, target_column,
                                                                            label_index, max_epochs=1)
    elif model_name == 'linear':
        model, val_performance, test_performance, _ = run_single_step_model('linear', single_step_window,
                                                                            wide_window, target_column,
                                                                            max_epochs=max_epochs,
                                                                            optimizer=optimizer)
    elif model_name == 'fc':
        model, val_performance, test_performance, _ = run_single_step_model('dense', single_step_window, wide_window,
                                                                            target_column, max_epochs=max_epochs)
    elif model_name == 'mulitstep-fc':
        model, val_performance, test_performance, _ = run_single_step_model('flattened dense',
                                                                            conv_window_gen,
                                                                            conv_window_gen,
                                                                            target_column,
                                                                            max_epochs=max_epochs,
                                                                            optimizer=optimizer)
    elif model_name == 'cnn':
        model, val_performance, test_performance, _ = run_single_step_model('cnn', conv_window_gen, wide_conv_window,
                                                                            target_column, conv_width=history_length,
                                                                            max_epochs=max_epochs,
                                                                            optimizer=optimizer)
    elif model_name == 'lstm':
        model, val_performance, test_performance, _ = run_single_step_model('lstm', conv_window_gen, wide_window,
                                                                            target_column, max_epochs=max_epochs)
    elif model_name == 'gru':
        model, val_performance, test_performance, _ = run_single_step_model('gru', conv_window_gen,
                                                                            wide_window, target_column,
                                                                            max_epochs=max_epochs,
                                                                            optimizer=optimizer)
    return model, val_performance, test_performance


def run_all_single_step_prediction_models(train_df, val_df, test_df, target_column, label_length,
                                          history_length=HISTORY_LENGTH, batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS,
                                          optimizer=None, val_percentage=None, test_percentage=None):
    single_step_window, wide_window, conv_window_gen, wide_conv_window = prepare_generators(train_df, val_df, test_df,
                                                                                            target_column, label_length,
                                                                                            history_length=history_length,
                                                                                            batch_size=batch_size,
                                                                                            val_percentage=val_percentage,
                                                                                            test_percentage=test_percentage, )
    if type(train_df) is dict:
        feature_names = train_df[list(train_df.keys())[0]].columns
    else:
        feature_names = train_df.columns
    label_index = feature_names.tolist().index(target_column)

    val_performance = {}
    performance = {}
    _, val_performance['Baseline'], performance['Baseline'], _ = run_single_step_model('baseline', single_step_window,
                                                                                       wide_window, target_column,
                                                                                       label_index, max_epochs=1)
    linear, val_performance['Linear'], performance['Linear'], _ = run_single_step_model('linear', single_step_window,
                                                                                        wide_window, target_column,
                                                                                        max_epochs=max_epochs,
                                                                                        optimizer=optimizer)

    linear.plot_weights(feature_names)

    _, val_performance['FC'], performance['FC'], _ = run_single_step_model('dense', single_step_window, wide_window,
                                                                           target_column, max_epochs=max_epochs)
    _, val_performance['Multistep-FC'], performance['Multistep-FC'], _ = run_single_step_model('flattened dense',
                                                                                               conv_window_gen,
                                                                                               conv_window_gen,
                                                                                               target_column,
                                                                                               max_epochs=max_epochs,
                                                                                               optimizer=optimizer)
    _, val_performance['CNN'], performance['CNN'], _ = run_single_step_model('cnn', conv_window_gen, wide_conv_window,
                                                                             target_column, conv_width=history_length,
                                                                             max_epochs=max_epochs,
                                                                             optimizer=optimizer)
    _, val_performance['LSTM'], performance['LSTM'], _ = run_single_step_model('lstm', conv_window_gen, wide_window,
                                                                               target_column, max_epochs=max_epochs)
    gru_model, val_performance['GRU'], performance['GRU'], _ = run_single_step_model('gru', conv_window_gen,
                                                                                     wide_window, target_column,
                                                                                     max_epochs=max_epochs,
                                                                                     optimizer=optimizer)

    metric_name = 'mean_absolute_error'
    metric_index = gru_model.metrics_names.index(metric_name)
    display_utils.plot_keras_models_performance(val_performance, performance, metric_index, metric_name, target_column)

    metric_name = 'root_mean_squared_error'
    metric_index = gru_model.metrics_names.index(metric_name)
    display_utils.plot_keras_models_performance(val_performance, performance, metric_index, metric_name, target_column)

    return val_performance, performance


def get_multi_step_prediction_model(model_name, train_df, val_df, test_df, target_column, in_steps=5, out_steps=5,
                                    batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS, optimizer=None,
                                    val_percentage=None, test_percentage=None):
    if type(train_df) is dict:
        feature_names = train_df[list(train_df.keys())[0]].columns
    else:
        feature_names = train_df.columns
    num_features = len(feature_names.tolist())

    multi_window = multi_step_window_generator(train_df, val_df, test_df, target_column,
                                               in_steps=in_steps, out_steps=out_steps,
                                               batch_size=batch_size,
                                               val_percentage=val_percentage,
                                               test_percentage=test_percentage, )
    print('Input shape:', multi_window.example[0].shape)
    print(multi_window)
    multi_window.plot(plot_col=target_column)

    label_index = feature_names.tolist().index(target_column)
    output_features = 1

    if model_name == 'last':
        model, val_performance, test_performance, _ = run_multi_step_model('lastbaseline', multi_window,
                                                                           prediction_length=out_steps,
                                                                           target_column=target_column,
                                                                           output_column_id=label_index,
                                                                           max_epochs=1)
    elif model_name == 'repeat':
        model, val_performance, test_performance, _ = run_multi_step_model('repeatbaseline',
                                                                           multi_window,
                                                                           target_column=target_column,
                                                                           output_column_id=label_index,
                                                                           max_epochs=1)
    elif model_name == 'linear':
        model, val_performance, test_performance, _ = run_multi_step_model('linear', multi_window,
                                                                           prediction_length=out_steps,
                                                                           num_output_features=output_features,
                                                                           target_column=target_column,
                                                                           max_epochs=max_epochs,
                                                                           optimizer=optimizer)
    elif model_name == 'fc':
        model, val_performance, test_performance, _ = run_multi_step_model('dense', multi_window,
                                                                           prediction_length=out_steps,
                                                                           num_output_features=output_features,
                                                                           target_column=target_column,
                                                                           max_epochs=max_epochs, optimizer=optimizer)
    elif model_name == 'cnn':
        conv_width = 3
        model, val_performance, test_performance, _ = run_multi_step_model('cnn', multi_window,
                                                                           prediction_length=out_steps,
                                                                           num_output_features=output_features,
                                                                           target_column=target_column,
                                                                           conv_width=conv_width,
                                                                           max_epochs=max_epochs, optimizer=optimizer)
    elif model_name == 'lstm':
        model, val_performance, test_performance, _ = run_multi_step_model('lstm', multi_window,
                                                                           prediction_length=out_steps,
                                                                           num_output_features=output_features,
                                                                           target_column=target_column,
                                                                           memory_units=32,
                                                                           max_epochs=max_epochs,
                                                                           optimizer=optimizer)
    elif model_name == 'arlstm':
        model, val_performance, test_performance, _ = run_multi_step_model('arlstm',
                                                                           multi_window,
                                                                           prediction_length=out_steps,
                                                                           num_output_features=output_features,
                                                                           target_column=target_column,
                                                                           output_column_id=label_index,
                                                                           memory_units=32,
                                                                           num_features=num_features,
                                                                           max_epochs=max_epochs,
                                                                           optimizer=optimizer)
        model.show_dimensions(multi_window)

    return model, val_performance, test_performance


def run_all_multi_step_prediction_models(train_df, val_df, test_df, target_column, in_steps=5, out_steps=5,
                                         batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS, optimizer=None,
                                         val_percentage=None, test_percentage=None):
    if type(train_df) is dict:
        feature_names = train_df[list(train_df.keys())[0]].columns
    else:
        feature_names = train_df.columns

    num_features = len(feature_names.tolist())
    multi_window = multi_step_window_generator(train_df, val_df, test_df, target_column,
                                               in_steps=in_steps, out_steps=out_steps,
                                               batch_size=batch_size,
                                               val_percentage=val_percentage,
                                               test_percentage=test_percentage, )
    print('Input shape:', multi_window.example[0].shape)
    print(multi_window)
    multi_window.plot(plot_col=target_column)

    label_index = feature_names.tolist().index(target_column)
    output_features = 1

    multi_val_performance = {}
    multi_performance = {}

    _, multi_val_performance['Last'], multi_performance['Last'], _ = run_multi_step_model('lastbaseline', multi_window,
                                                                                          prediction_length=out_steps,
                                                                                          target_column=target_column,
                                                                                          output_column_id=label_index,
                                                                                          max_epochs=1)

    _, multi_val_performance['Repeat'], multi_performance['Repeat'], _ = run_multi_step_model('repeatbaseline',
                                                                                              multi_window,
                                                                                              target_column=target_column,
                                                                                              output_column_id=label_index,
                                                                                              max_epochs=1)

    _, multi_val_performance['Linear'], multi_performance['Linear'], _ = run_multi_step_model('linear', multi_window,
                                                                                              prediction_length=out_steps,
                                                                                              num_output_features=output_features,
                                                                                              target_column=target_column,
                                                                                              max_epochs=max_epochs,
                                                                                              optimizer=optimizer)

    _, multi_val_performance['FC'], multi_performance['FC'], _ = run_multi_step_model('dense', multi_window,
                                                                                      prediction_length=out_steps,
                                                                                      num_output_features=output_features,
                                                                                      target_column=target_column,
                                                                                      max_epochs=max_epochs,
                                                                                      optimizer=optimizer)
    conv_width = 3
    _, multi_val_performance['CNN'], multi_performance['CNN'], _ = run_multi_step_model('cnn', multi_window,
                                                                                        prediction_length=out_steps,
                                                                                        num_output_features=output_features,
                                                                                        target_column=target_column,
                                                                                        conv_width=conv_width,
                                                                                        max_epochs=max_epochs,
                                                                                        optimizer=optimizer)
    _, multi_val_performance['LSTM'], multi_performance['LSTM'], _ = run_multi_step_model('lstm', multi_window,
                                                                                          prediction_length=out_steps,
                                                                                          num_output_features=output_features,
                                                                                          target_column=target_column,
                                                                                          memory_units=32,
                                                                                          max_epochs=max_epochs,
                                                                                          optimizer=optimizer)

    arlstm, multi_val_performance['AR_LSTM'], multi_performance['AR_LSTM'], _ = run_multi_step_model('arlstm',
                                                                                                     multi_window,
                                                                                                     prediction_length=out_steps,
                                                                                                     num_output_features=output_features,
                                                                                                     target_column=target_column,
                                                                                                     output_column_id=label_index,
                                                                                                     memory_units=32,
                                                                                                     num_features=num_features,
                                                                                                     max_epochs=max_epochs,
                                                                                                     optimizer=optimizer)
    arlstm.show_dimensions(multi_window)

    metric_name = 'mean_absolute_error'
    metric_index = arlstm.metrics_names.index(metric_name)
    display_utils.plot_keras_models_performance(multi_val_performance, multi_performance, metric_index, metric_name,
                                                target_column)

    metric_name = 'root_mean_squared_error'
    metric_index = arlstm.metrics_names.index(metric_name)
    display_utils.plot_keras_models_performance(multi_val_performance, multi_performance, metric_index, metric_name,
                                                target_column)

    multi_performance = pd.DataFrame(multi_performance).transpose()
    multi_performance.rename(columns={0: 'MSELoss', 1: 'MSE', 2: 'MAE', 3: 'RMSE', 4: 'MAPE'}, inplace=True)
    multi_performance.drop(columns=['MSELoss'], inplace=True)

    multi_val_performance = pd.DataFrame(multi_val_performance).transpose()
    multi_val_performance.rename(columns={0: 'MSELoss', 1: 'MSE', 2: 'MAE', 3: 'RMSE', 4: 'MAPE'}, inplace=True)
    multi_val_performance.drop(columns=['MSELoss'], inplace=True)

    return multi_val_performance, multi_performance


################################################################################


def run_multiple_arima_models(df_train, df_test, pred_length, has_optimization=False, has_reverse=False,
                              orig_df_train=None, orig_df_test=None):
    from libs.foxutils.utils import arima_models
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    target_vals = df_train

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        print('Manual ARIMA')
        best_manual_arima_model = arima_models.get_manual_arima(target_vals)

        print('\nStepwise Auto ARIMA')
        best_autoarima_model = arima_models.get_stepwise_auto_arima(target_vals)

        if has_optimization:
            print('\nOptimized SARIMAX')
            best_sarima_model = arima_models.get_optimized_sarima(target_vals)
        else:
            best_sarima_model = []

    target_vals = df_test
    predictions = pd.DataFrame({
        'Pred_Manual_ARIMA': arima_models.forecast_values(best_manual_arima_model, pred_length),
        'Pred_Auto_ARIMA': arima_models.predict_values(best_autoarima_model, pred_length),
    })

    if has_optimization:
        predictions['Pred_Optimized_SARIMAX'] = arima_models.forecast_values(best_sarima_model, pred_length)

    if has_reverse:
        predictions_2 = pd.DataFrame(columns=predictions.columns)
        for x in predictions.columns:
            print(x)
            predictions_2[x] = arima_models.reverse_diff(predictions[x].values, orig_df_test.iloc[0])

        predictions = predictions_2.copy()
    predictions.reset_index(drop=True, inplace=True)

    if has_reverse:
        predictions['Actual'] = orig_df_test.values
    else:
        predictions['Actual'] = target_vals[:pred_length].values

    predictions.index = range(len(df_train), len(df_train) + len(predictions))

    if has_reverse:
        actual = pd.DataFrame({
            'Actual': orig_df_train
        })
    else:
        actual = pd.DataFrame({
            'Actual': df_train
        })

    actual['Pred_Manual_ARIMA'] = np.nan
    actual['Pred_Auto_ARIMA'] = np.nan

    if has_optimization:
        actual['Pred_Optimized_SARIMAX'] = np.nan

    predictions_and_actual = pd.concat([actual, predictions])
    predictions_and_actual = predictions_and_actual.reindex(sorted(predictions_and_actual.columns), axis=1)

    predictions.reset_index(inplace=True, drop=True)
    predictions = predictions.reindex(sorted(predictions.columns), axis=1)

    df_results = train_functionalities.get_error_metrics(predictions['Actual'], predictions['Pred_Manual_ARIMA'])
    df_results.index = ['Manual_ARIMA']

    res = train_functionalities.get_error_metrics(predictions['Actual'], predictions['Pred_Auto_ARIMA'])
    res.index = ['Pred_Auto_ARIMA']
    df_results = pd.concat([df_results, res])

    if has_optimization:
        res = train_functionalities.get_error_metrics(predictions['Actual'], predictions['Optimized_SARIMAX'])
        res.index = ['Optimized_SARIMAX']
        df_results = pd.concat([df_results, res])

    predictions_and_actual.plot(legend=True, figsize=(20, 8))
    predictions.plot(legend=True, figsize=(20, 8))

    return best_manual_arima_model, best_autoarima_model, best_sarima_model, df_results
