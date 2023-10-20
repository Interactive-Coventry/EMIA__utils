from sklearn.preprocessing import MinMaxScaler
import math
import os
import numpy as np
import pandas as pd
from os.path import join as pathjoin
from tensorflow import stack
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
from pathlib import Path
from libs.foxutils.utils import core_utils, train_functionalities

filename_format = 'ts_file{}.pkl'


def create_ts_files(dataset,
                    start_index,
                    end_index,
                    history_length,
                    step_size,
                    target_step,
                    num_rows_per_file,
                    data_folder):
    # history_length = values to loockback at
    # step_size = step inbetween loockback values
    # target_step = how many steps later after last loockback value to predict
    assert step_size > 0
    assert start_index >= 0

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    time_lags = sorted(range(target_step + 1, target_step + history_length + 1, step_size), reverse=True)
    col_names = [f'x_lag{i}' for i in time_lags] + ['y']
    start_index = start_index + history_length
    if end_index is None:
        end_index = len(dataset) - target_step

    rng = range(start_index, end_index)
    num_rows = len(rng)
    num_files = math.ceil(num_rows / num_rows_per_file)

    # for each file.
    print(f'Creating {num_files} files.')
    for i in range(num_files):
        filename = f'{data_folder}/ts_file{i}.pkl'

        if i % 10 == 0:
            print(f'{filename}')

        # get the start and end indices.
        ind0 = i * num_rows_per_file + start_index
        ind1 = min(ind0 + num_rows_per_file, end_index)
        data_list = []

        # j in the current timestep. Will need j-n to j-1 for the history. And j + target_step for the target.
        for j in range(ind0, ind1):
            indices = range(j - 1, j - history_length - 1, -step_size)
            data = dataset[sorted(indices) + [j + target_step]]

            # append data to the list.
            data_list.append(data)

        df_ts = pd.DataFrame(data=data_list, columns=col_names)
        df_ts.to_pickle(filename)

    return len(col_names) - 1


#
# So we can handle loading the data in chunks from the hard drive instead of having to load everything into memory.
#
# The reason we want to do this is so we can do custom processing on the data that we are feeding into the LSTM.
# LSTM requires a certain shape and it is tricky to get it right.
#
class TimeSeriesLoader:
    def __init__(self, ts_folder, filename_format):
        self.ts_folder = ts_folder

        # find the number of files.
        i = 0
        file_found = True
        while file_found:
            filename = self.ts_folder + '/' + filename_format.format(i)
            file_found = os.path.exists(filename)
            if file_found:
                i += 1

        self.num_files = i
        self.files_indices = np.arange(self.num_files)
        self.shuffle_chunks()

    def num_chunks(self):
        return self.num_files

    def get_chunk(self, idx):
        assert (idx >= 0) and (idx < self.num_files)

        ind = self.files_indices[idx]
        filename = self.ts_folder + '/' + filename_format.format(ind)
        df_ts = pd.read_pickle(filename)
        num_records = len(df_ts.index)

        features = df_ts.drop('y', axis=1).values
        target = df_ts['y'].values

        # reshape for input into LSTM. Batch major format.
        features_batchmajor = np.array(features).reshape(num_records, -1, 1)
        return features_batchmajor, target

    # this shuffles the order the chunks will be outputted from get_chunk.
    def shuffle_chunks(self):
        np.random.shuffle(self.files_indices)


def train_and_validate_lstm(train_vals, test_vals, batch_size=32, num_epochs=20, history_length=10, step_size=1,
                            target_step=5, experiment_name='bus_arrivals', version=1):
    num_rows_per_file = 50
    data_folder = pathjoin('../temp_files', experiment_name, 'lstm', 'train')

    # Scaled to work with Neural networks.
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_vals = scaler.fit_transform(train_vals.reshape(-1, 1)).reshape(-1, )
    num_timesteps = create_ts_files(train_vals, 0, None, history_length, step_size, target_step, num_rows_per_file,
                                    data_folder)

    tss = TimeSeriesLoader(data_folder, filename_format)

    ts_inputs = Input(shape=(num_timesteps, 1))
    # units=10 -> The cell and hidden states will be of dimension 10.
    #             The number of parameters that need to be trained = 4*units*(units+2)
    x = LSTM(units=history_length)(ts_inputs)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='linear')(x)
    lstm_model = Model(inputs=ts_inputs, outputs=outputs)

    lstm_model.compile(loss='mean_squared_error',
                       optimizer='adam',
                       metrics=['mean_absolute_error'])
    print(lstm_model.summary())

    NUM_CHUNKS = tss.num_chunks()

    checkpoint_dir = pathjoin(core_utils.models_dir, 'lstm', experiment_name, 'lstm-' + str(version), 'training', 'version')
    checkpoint_path = pathjoin(Path(core_utils.increment_path(checkpoint_dir, exist_ok=False)), 'cp-{epoch:04d}.ckpt')

    # Create a callback that saves the model's weights
    cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                  verbose=1,
                                  save_weights_only=True,
                                  save_freq=5 * batch_size
                                  )

    for epoch in range(num_epochs):
        print('epoch #{}'.format(epoch))
        for i in range(NUM_CHUNKS):
            X, y = tss.get_chunk(i)

            # model.fit does train the model incrementally. ie. Can call multiple times in batches.
            # https://github.com/keras-team/keras/issues/4446
            lstm_model.fit(x=X, y=y, batch_size=batch_size, callbacks=[cp_callback])

        # shuffle the chunks so they're not in the same order next time around.
        tss.shuffle_chunks()

    data_folder_test = pathjoin('../temp_files', 'lstm', 'val')
    test_vals = scaler.transform(test_vals.reshape(-1, 1)).reshape(-1, )
    num_timesteps = create_ts_files(test_vals, 0, None, history_length, step_size, target_step, num_rows_per_file,
                                    data_folder_test)
    # If we assume that the validation dataset can fit into memory we can do this.
    df_val_ts = pd.read_pickle(data_folder_test + '/ts_file0.pkl')

    features = df_val_ts.drop('y', axis=1).values
    features_arr = np.array(features)

    # reshape for input into LSTM. Batch major format.
    num_records = len(df_val_ts.index)
    features_batchmajor = features_arr.reshape(num_records, -1, 1)

    y_pred = lstm_model.predict(features_batchmajor).reshape(-1, )
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1, )

    y_act = df_val_ts['y'].values
    y_act = scaler.inverse_transform(y_act.reshape(-1, 1)).reshape(-1, )

    print('validation mean squared error: {}'.format(mean_squared_error(y_act, y_pred)))

    # baseline
    df_val_ts.columns[0]
    y_pred_baseline = df_val_ts[df_val_ts.columns[0]].values
    y_pred_baseline = scaler.inverse_transform(y_pred_baseline.reshape(-1, 1)).reshape(-1, )
    print('validation baseline mean squared error: {}'.format(mean_squared_error(y_act, y_pred_baseline)))

    df_preds = pd.DataFrame({'Actual': y_act, 'Pred': y_pred})
    df_results = train_functionalities.get_error_metrics(df_preds['Actual'], df_preds['Pred'])
    df_results.index = ['lstm']
    return lstm_model, df_preds, df_results


def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
        labels = stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels


class WindowGenerator:
    def __init__(self, input_width, label_width, shift, data_df, label_columns=None):
        # Store the raw data.
        self.data_df = data_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(data_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
        self.split_window = split_window

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])


