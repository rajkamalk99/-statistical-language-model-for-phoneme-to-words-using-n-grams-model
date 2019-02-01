import os
from difflib import get_close_matches 
import nltk

# fucnction for creating the list containing the phonetic trancription and corresponding text
def extract_data(phn_file, wrd_file): 
        bigram_list = []
        with open(wrd_file, "r") as f:
                text_contents = f.readlines()
        with open(phn_file, "r") as f:
                phoneme_contents = f.readlines()
        for text_line in text_contents:
                text_line_list = text_line.split(" ")
                text_lb = text_line_list[0]
                text_ub = text_line_list[1]
                text    = text_line_list[2]
                text = text[:-1]
                phoneme_string = ""
                flag = 0
                for phoneme_line in phoneme_contents:
                        phoneme_line_list = phoneme_line.split(" ")
                        if flag == 1:
                                phoneme_sub_string = str(phoneme_line_list[2])
                                phoneme_sub_string = phoneme_sub_string[:-1]
                                phoneme_string += str(phoneme_sub_string)
                                if phoneme_line_list[1] == text_ub:
                                  flag = 0
                        if phoneme_line_list[0] == text_lb:
                                phoneme_sub_string = str(phoneme_line_list[2])
                                phoneme_sub_string = phoneme_sub_string[:-2]
                                phoneme_string += str(phoneme_sub_string)
                                flag = 1
                bigram_list.append([phoneme_string, text])
        return bigram_list

# in case it's executed in a different machine just change these base directory paths and that's it.

train_dir = "/home/raj/Documents/AI/novaturient/TIMIT/data/lisa/data/timit/raw/TIMIT/TRAIN/"
test_dir = "/home/raj/Documents/AI/novaturient/TIMIT/data/lisa/data/timit/raw/TIMIT/TEST/"
# the main function fetches the phonetic and corresponding text files and sends to extract_data() function
def main(base_dir):
        bigram = []
        dr_files = os.listdir(base_dir)
        for dr in dr_files:
                gender_files_path = str(base_dir) + str(dr) + "/"
                gender_files = os.listdir(gender_files_path)
                for gender_file in gender_files:
                        speech_files_path = str(gender_files_path) + str(gender_file) + "/"
                        speech_files = os.listdir(speech_files_path)
                        wrd = []
                        phn = []
                        for file_item in speech_files:
                                if file_item[-3:] == "PHN":
                                        phn.append(file_item)
                        for phn_item in phn:
                                name = phn_item[:-4]
                                wrd_name = str(name) + ".WRD"
                                bigram.append(extract_data(str(speech_files_path) + str(phn_item), str(speech_files_path)+str(wrd_name)))
        return bigram

# function for creating individual list with phonetic trancription and text
def preprocess(pre_list):
        phonetic = []
        text = []
        for dr in pre_list:
                for gender in dr:
                        phonetic.append(gender[0])
                        text.append(gender[1])

        return phonetic, text


# function for predicting the most matched words for the given test set
def predict_word(train_phonetic, test_phonetic, train_text):
        predict_final = []
        final = []
        for phoneme in test_phonetic:
                # get_close_matches function returnes the most matched word as n is set to 2 it returns two most matched words
                word = get_close_matches(phoneme, train_phonetic, n=2)
                if len(word) != 0:
                        index = train_phonetic.index(word[0])
                        text_word = train_text[index]
                        predict_final.append(text_word)
                        final.append([phoneme, text_word])
                        # print(phoneme, text_word)
                else:
                        predict_final.append(None)
        return final, predict_final

train_list = main(train_dir)
test_list  = main(test_dir)
train_phonetic, train_text = preprocess(train_list)
test_phonetic, test_text = preprocess(test_list)
final, predict_final = predict_word(train_phonetic, test_phonetic, train_text)

orginal = []
predicted = []
for index in range(len(predict_final)):
        if predict_final[index] != None:
                predicted.append(predict_final[index])
                orginal.append(test_text[index])
wer = []
for i in range(len(orginal)):
        # using the edit_distance() function for calculating the levenshtein distance 
        dist = nltk.edit_distance(orginal[i],predicted[i])
        # calculating the word error rate and appending it to a list
        wer.append(dist/len(orginal[i]))

# calculating the average word error rate
mean_wer = sum(wer)/len(wer)
print("Wer is ",mean_wer)
