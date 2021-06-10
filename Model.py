import numpy as np
import pandas as pd
import tensorflow as tf
import time
np.random.seed(1004)
df = pd.read_csv("separate_SGG_NS.csv",index_col=[0])
#df = pd.read_csv("separate_SGG_DG.csv",index_col=[0])
df_gangnam = df.loc[df.SGG == '강남']
df_gangbuk = df.loc[df.SGG == '강북']
df_gangnam.pop('SGG')
df_gangbuk.pop('SGG')
df=df_gangbuk  ####### 여기 수정해서 돌리기

# df_daro = df.loc[df.SGG == '대로']
# df_gu = df.loc[df.SGG == '구']
# df_daro.pop('SGG')
# df_gu.pop('SGG')
# df=df_gu
# print(df_gu)

date_time = pd.to_datetime(df.pop('DATE'), format='%Y.%m.%d %H:%M:%S')
##########################################################################################################################
##########################################################################################################################
# 데이터 분할 ## 여기 숫자 수정예정!!
column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]

##########################################################################################################################
## WindowGenerator 클래스를 생성 / train, eval, test 데이터 프레임을 입력으로 사용 / 이 데이터 프레임은 나중에 창의 tf.data.Dataset으로 변환
class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in enumerate(train_df.columns)}
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

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

##########################################################################################################################
def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels

WindowGenerator.split_window = split_window
##########################################################################################################################
# tf.data.Dataset 만들기
def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.preprocessing.timeseries_dataset_from_array( #compat.v1.
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=False,
      batch_size=32,)
  ds = ds.map(self.split_window)
  return ds


MAX_EPOCHS = 500 #############확인!

def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=patience,mode='min')
  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(learning_rate= 0.001),
                metrics=[tf.metrics.MeanAbsoluteError(), tf.metrics.RootMeanSquaredError()])
                # MAE: 평균 절대 오차, RMSE: 제곱근 평균 제곱 오차
  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  print(model.summary())

  return history

##########################################################################################################################
##########################################################################################################################
WindowGenerator.make_dataset = make_dataset

@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example
wide_window = WindowGenerator(input_width=365, label_width=1, shift=0, label_columns=['SO2']) ### label_column 수정하고

gru_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.GRU(64, return_sequences=True),
    tf.keras.layers.GRU(32, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=num_features)
])

simple_RNN_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.SimpleRNN(64, return_sequences=True),
    tf.keras.layers.SimpleRNN(32, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    # Shap thee => [batch, time, features]
    tf.keras.layers.Dense(units=num_features)
])

bilstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dropout(0.2),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=num_features)
])

lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=num_features)
])

proposed_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(3, 5, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(units=num_features, activation='relu'),
])

duration={}
print("---------------- GRU ----------------")
start=time.time()
history1 = compile_and_fit(gru_model, wide_window)
end=time.time()
duration['GRU']=end-start
print("---------------- simpleRNN ----------------")
start=time.time()
history2 = compile_and_fit(simple_RNN_model, wide_window)
end=time.time()
duration['RNN']=end-start
print("---------------- BiLSTM ----------------")
start=time.time()
history3 = compile_and_fit(bilstm_model, wide_window)
end=time.time()
duration['BiLSTM']=end-start
print("---------------- LSTM ----------------")
start=time.time()
history4 = compile_and_fit(lstm_model, wide_window)
end=time.time()
duration['LSTM']=end-start
print("---------------- PROPOSED ----------------")
start=time.time()
history5 = compile_and_fit(proposed_model, wide_window)
end=time.time()
duration['PROPOSED']=end-start

result_duration=pd.DataFrame(duration,index=["Time"])
print(result_duration)


print("---------------- VALIDATION PERFORMANCE ----------------")
val_performance = {}
performance = {}
val_performance['LSTM'] = lstm_model.evaluate(wide_window.val)
performance['LSTM'] = lstm_model.evaluate( wide_window.test)
val_performance['BiLSTM'] = bilstm_model.evaluate(wide_window.val)
performance['BiLSTM'] = bilstm_model.evaluate( wide_window.test)
val_performance['RNN'] = simple_RNN_model.evaluate(wide_window.val)
performance['RNN'] = simple_RNN_model.evaluate( wide_window.test)
val_performance['GRU'] = gru_model.evaluate(wide_window.val)
performance['GRU'] = gru_model.evaluate( wide_window.test)
val_performance['PROPOSED'] = proposed_model.evaluate(wide_window.val)
performance['PROPOSED'] = proposed_model.evaluate( wide_window.test)

results=pd.DataFrame(performance,index=["MSE","MAE","RMSE"])
print(results)
results.to_csv("a1.csv")