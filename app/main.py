# !/usr/bin/python
# -*-coding:utf-8-*-

# /Users/shanxiliao/Documents/python-workspace/tagging/env/lib/python3.7/site-packages/kivy/data/fonts
# mv Roboto-Regular.ttf Roboto-Regular0.ttf
# cp /Library/Fonts/Arial\ Unicode.ttf ./Roboto-Regular.ttf
import threading

from kivy.app import App
# from kivy.uix.floatlayout import FloatLayout
# from kivy.clock import Clock
# from kivy.properties import partial
from kivy.uix.label import Label
from kivy.uix.button import Button
# from functools import partial
from kivy.uix.boxlayout import BoxLayout
# from kivy.lang import Builder
# from kivy.uix.recycleview import RecycleView
# from kivy.base import runTouchApp
from kivy.uix.textinput import TextInput
# from kivy.clock import Clock
from kivy.uix.gridlayout import GridLayout
from datetime import datetime
import pandas as pd
import tensorflow as tf
from kivy.uix.togglebutton import ToggleButton
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.layers import Dropout, GlobalMaxPool1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import hanlp
import numpy as np

__vserion__ = '1.0.0'


class SegmentWord:
    def __init__(self):
        self.tokenizer = hanlp.load('PKU_NAME_MERGED_SIX_MONTHS_CONVSEG')

    def seg(self, text):
        seg_list = self.tokenizer(text)
        word_list = []
        for word_ in seg_list:
            word_list.append(word_)
        word_str = ' '.join(word_list)
        return word_str

    def batch_seg(self, file_path=None):
        df = pd.read_csv(file_path)

        seg_list_all = []
        for i in df['store_prod_name']:
            seg_list = self.tokenizer(i)
            word_list = []
            for word_ in seg_list:
                word_list.append(word_)
            word_str = ' '.join(word_list)
            seg_list_all.append(word_str)

        return seg_list_all


class Predict:
    def __init__(self, **kwargs):
        # from tf.keras.models import load_model
        self.model = tf.keras.models.load_model("model.hdf5")
        self.training_dataset = 'training_dataset.csv'
        self.training_segment_dataset = 'training_segment_dataset.csv'
        self.segmentWord = SegmentWord()
        self.label_map = {4: '餐飲', 0: '按摩', 2: '美顏/造型', 3: '運動休閒', 1: '旅遊'}
        self.console = kwargs['console']
        self.logger = kwargs['logger']
        self.logger(self.console, f'標籤清單: {self.label_map}')
        self.result_show_place = kwargs['result_show_place']

    def do_predict(self, content):
        self.logger(self.console, f'預測內容: {content}')
        seg_word = self.segmentWord.seg(content)
        content = seg_word
        self.logger(self.console, f'斷詞結果: {content}')
        self.logger(self.console, '開始預測...')

        max_words = 1000
        max_len = 500
        df_seg = pd.read_csv(self.training_segment_dataset)
        df = pd.read_csv(self.training_dataset)
        df = df[['store_prod_name', 'parent_tag']]
        df_all = pd.concat([df, df_seg], axis=1)
        # from keras.preprocessing.text import Tokenizer
        tok = tf.keras.preprocessing.text.Tokenizer(num_words=max_words)
        tok.fit_on_texts(df_all.seg_word.to_numpy())
        text = [content]
        test_seq = tok.texts_to_sequences(text)
        # from keras.preprocessing import sequence
        test_seq_mat = tf.keras.preprocessing.sequence.pad_sequences(test_seq, maxlen=max_len)
        test_seq_pre = self.model.predict_classes(test_seq_mat)
        result = [f'{t} => {self.label_map[test_seq_pre[i]]}' for i, t in enumerate(text)]
        self.logger(self.console, f'預測結果:{test_seq_pre[0]}, {result[0]}')
        self.result_show_place.text = self.label_map[test_seq_pre[0]]

    def batch_predict(self, file_path=None):
        try:
            self.logger(self.console, f'批次預測檔案路徑: {file_path}')
            seg_word_list = self.segmentWord.batch_seg(file_path)

            max_words = 1000
            max_len = 500
            df_seg = pd.read_csv(self.training_segment_dataset)
            df = pd.read_csv(self.training_dataset)
            df = df[['store_prod_name', 'parent_tag']]
            df_all = pd.concat([df, df_seg], axis=1)
            tok = tf.keras.preprocessing.text.Tokenizer(num_words=max_words)
            tok.fit_on_texts(df_all.seg_word.to_numpy())

            text = seg_word_list
            test_seq = tok.texts_to_sequences(text)
            test_seq_mat = sequence.pad_sequences(test_seq,maxlen=max_len)
            test_seq_pre = self.model.predict_classes(test_seq_mat)
            predict_list = list()
            seg_wrod_list = list()

            for i, t in enumerate(text):
                self.logger(self.console, f'{t} => {self.label_map[test_seq_pre[i]]}')
                predict_list.append(self.label_map[test_seq_pre[i]])
                seg_wrod_list.append(t)

            predict_df = pd.DataFrame(list(zip(predict_list, seg_wrod_list)), columns=['Tag','Seg_word'])
            predict_df.to_csv('batch_predict_result.csv', index=False)

            self.logger(self.console, '批次貼標結果: batch_predict_result.csv')

        except Exception as ex:
            self.logger(self.console, f'error:{ex}')

class Train:
    def __init__(self, **kwargs):
        # self.model = tf.keras.models.load_model("model.hdf5")
        self.training_dataset = 'training_dataset.csv'
        self.training_segment_dataset = 'training_segment_dataset.csv'
        self.console = kwargs['console']
        self.logger = kwargs['logger']

    def do_train(self, mode=None):
        try:
            self.logger(self.console, '準備開始訓練')
            df_seg = pd.read_csv(f'./{self.training_segment_dataset}')
            df = pd.read_csv(f'./{self.training_dataset}')
            df = df[['store_prod_name', 'parent_tag']]
            df_all = pd.concat([df, df_seg], axis=1)
            _y = df_all.parent_tag
            copy_y = _y

            le = LabelEncoder()
            _y = le.fit_transform(_y).reshape(-1, 1)

            label_map = dict()
            for idx, v in enumerate(_y):
                label_map.update({v[0]: copy_y[idx]})

            self.logger(self.console, f'{label_map}')

            ohe = OneHotEncoder()
            _y = ohe.fit_transform(_y).toarray()

            _x = df_all.seg_word.to_numpy()
            train_x, val_x, train_y, val_y = train_test_split(_x, _y, test_size=0.2, random_state=1, stratify=_y)
            max_words = 1000
            max_len = 500
            tok = Tokenizer(num_words=max_words)
            tok.fit_on_texts(_x)
            train_seq = tok.texts_to_sequences(train_x)
            train_seq_mat = sequence.pad_sequences(train_seq, maxlen=max_len)
            val_seq = tok.texts_to_sequences(val_x)
            val_seq_mat = sequence.pad_sequences(val_seq, maxlen=max_len)

            if mode == 'NEW':
                self.model = Sequential()
                self.model.add(layers.Embedding(max_words + 1, 64, input_length=max_len))
                self.model.add(layers.Bidirectional(layers.LSTM(
                    64,
                    dropout=0.4,
                    recurrent_dropout=0.2,
                    return_sequences=True,
                    kernel_initializer='he_normal')))

                self.model.add(layers.Dense(64, activation="relu", kernel_initializer='he_normal'))
                self.model.add(Dropout(0.4))
                self.model.add(GlobalMaxPool1D())
                self.model.add(layers.Dense(64, activation="relu", kernel_initializer='he_normal'))
                self.model.add(Dropout(0.4))
                self.model.add(layers.Dense(5, activation='softmax'))
                self.model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])
            else:
                self.model = tf.keras.models.load_model("model.hdf5")

            filepath = "model.hdf5"
            checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
            early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)
            callbacks_list = [
                checkpoint,
                early_stopping
            ]

            self.logger(self.console, '訓練中...')
            model_fit = self.model.fit(
                train_seq_mat,
                train_y,
                batch_size=128,
                epochs=200,
                validation_data=(val_seq_mat, val_y),
                callbacks=callbacks_list
            )

            self.logger(self.console, '完成訓練')
        except Exception as ex:
            self.logger(self.console, f'訓練發生錯誤: {ex}')


class Main(App):

    def log(self, instance, message, *args):
        if instance.text != '':
            instance.text = instance.text + f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n"
        else:
            instance.text = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n"

    def async_log(self, instance, message):
        t = threading.Thread(target=self.log, args=(instance, message))
        t.daemon = True
        t.start()

    def build(self):
        self.title = f'自動貼標機器人 {__vserion__}'

        root = GridLayout(cols=1)

        header_block = GridLayout(cols=3,
                                  row_default_height=50,
                                  row_force_default=True,
                                  size_hint_y=None,
                                  height=50)

        grid_block = GridLayout(cols=3)

        grid_block_1 = GridLayout(cols=1)
        grid_block_2 = GridLayout(cols=1)
        grid_block_3 = GridLayout(cols=1)

        group_component_a = GridLayout(cols=1)
        group_component_b = GridLayout(cols=1)
        group_component_c = GridLayout(cols=1)
        group_component_d = GridLayout(rows=2)

        header_block.add_widget(Label(text="單筆預測", color=(.8, .9, 0, 1)))

        self.component_1 = component_1 = BoxLayout(orientation='horizontal')
        component_1.add_widget(Label(text="商店名稱", color=(1, 0.757, 0.145, 1)))
        component_1.store_name = TextInput(hint_text='商店名稱')
        component_1.add_widget(component_1.store_name)
        self.store_name = component_1.store_name

        self.component_2 = component_2 = BoxLayout(orientation='horizontal')
        component_2.add_widget(Label(text="商品名稱", color=(1, 0.757, 0.145, 1)))
        component_2.prod_name = TextInput(hint_text='商品名稱')
        component_2.add_widget(component_2.prod_name)
        self.prod_name = component_2.prod_name

        self.component_2_1 = component_2_1 = BoxLayout(orientation='horizontal')
        component_2_1.add_widget(Label(text="預測結果", color=(1, 0.757, 0.145, 1)))
        component_2_1.predict_tag = TextInput()
        component_2_1.add_widget(component_2_1.predict_tag)
        self.predict_tag = component_2_1.predict_tag

        self.component_3 = component_3 = BoxLayout(orientation='vertical')
        component_3.tagging_btn = Button(text="預測貼標")
        component_3.tagging_btn.bind(on_press=self.do_tagging)
        component_3.add_widget(component_3.tagging_btn)

        self.compoent_4 = compoent_4 = BoxLayout()
        compoent_4.console = TextInput(multiline=True, readonly=True)
        compoent_4.add_widget(compoent_4.console)
        self.console = compoent_4.console

        # self.component_4_1 = component_4_1 = BoxLayout(orientation='vertical')
        clear_log_btn = Button(text="清除 log", size_hint_y=None, size_hint_x = None, width = 300)
        clear_log_btn.bind(on_press=self.clear_log)
        # component_4_1.add_widget(component_4_1.clear_log_btn)

        group_component_a.add_widget(component_1)
        group_component_a.add_widget(component_2)
        group_component_a.add_widget(component_2_1)

        header_block.add_widget(Label(text="模型預測", color=(.8, .9, 0, 1)))

        self.component_5 = component_5 = BoxLayout(orientation='vertical')
        component_5.train_btn = Button(text="訓練模型")
        component_5.train_btn.bind(on_press=self.do_training)
        component_5.add_widget(component_5.train_btn)

        self.component_6 = component_6 = BoxLayout(orientation='horizontal')
        component_6.add_widget(Label(text="檔案路徑", color=(1, 0.757, 0.145, 1)))
        component_6.train_data_file_path = TextInput(hint_text='檔案路徑')
        component_6.add_widget(component_6.train_data_file_path)
        self.train_data_file_path = component_6.train_data_file_path

        self.component_7 = component_7 = BoxLayout(orientation='horizontal')
        self.btn1 = btn1 = ToggleButton(text='訓練新模型', group='type')
        self.btn2 = btn2 = ToggleButton(text='訓練現存模型', group='type', state='down')

        component_7.add_widget(btn1)
        component_7.add_widget(btn2)

        group_component_b.add_widget(component_7)
        group_component_b.add_widget(component_6)
        group_component_b.add_widget(component_5)

        header_block.add_widget(Label(text="批次預測", color=(.8, .9, 0, 1)))

        self.component_7 = component_7 = BoxLayout(orientation='horizontal')
        component_7.add_widget(Label(text="檔案路徑", color=(1, 0.757, 0.145, 1)))
        component_7.predict_data_file_path = TextInput(hint_text='檔案路徑')
        component_7.add_widget(component_7.predict_data_file_path)
        self.predict_data_file_path = component_7.predict_data_file_path
        self.predict_data_file_path.text = 'batch_predict.csv'

        self.component_8 = component_8 = BoxLayout(orientation='vertical')
        component_8.batch_predict_btn = Button(text="批次預測貼標")
        component_8.batch_predict_btn.bind(on_press=self.do_batch_tagging)
        component_8.add_widget(component_8.batch_predict_btn)

        group_component_c.add_widget(component_7)
        group_component_c.add_widget(component_8)

        grid_block_1.add_widget(group_component_a)
        grid_block_1.add_widget(component_3)

        grid_block_2.add_widget(group_component_b)
        grid_block_3.add_widget(group_component_c)

        grid_block.add_widget(grid_block_1)
        grid_block.add_widget(grid_block_2)
        grid_block.add_widget(grid_block_3)

        root.add_widget(header_block)
        root.add_widget(grid_block)
        group_component_d.add_widget(clear_log_btn)
        group_component_d.add_widget(compoent_4)
        root.add_widget(group_component_d)
        return root

    def on_start(self, **kwargs):
        self.log(self.console, 'App loaded')
        try:
            self.predict = Predict(console=self.console, logger=self.async_log, result_show_place=self.predict_tag)
            self.log(self.console, 'Model loaded')
        except Exception as ex:
            self.log(self.console, f'Load model error: {ex}')

        try:
            self.train = Train(console=self.console, logger=self.async_log)
            self.log(self.console, 'Train tool loaded')
        except Exception as ex:
            self.log(self.console, f'Train tool loaded error: {ex}')

    def do_tagging(self, instance):
        text = f'{self.store_name.text} - {self.prod_name.text}'
        try:
            t = threading.Thread(target=self.predict.do_predict, args=(text,))
            t.daemon = True
            t.start()
        except Exception as ex:
            self.async_log(self.console, f'error: {ex}')

    def do_batch_tagging(self, instance):

        try:
            t = threading.Thread(target=self.predict.batch_predict, args=(self.predict_data_file_path.text,))
            t.daemon = True
            t.start()
        except Exception as ex:
            self.async_log(self.console, f'error: {ex}')

    def do_training(self, instance):

        mode = 'OLD'
        if self.btn1.state == 'down':
            mode = 'NEW'

        self.async_log(self.console, f'模式: {mode}')

        t = threading.Thread(target=self.train.do_train, args=(mode,))
        t.daemon = True
        t.start()

    def clear_log(self, instance):
        self.console.text = ''

Main().run()
