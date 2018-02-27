# -*- coding: utf-8 -*-
import os
import string
import argparse
import operator
from functools import partial
from difflib import SequenceMatcher

from audio.get_duration import get_durations
from text import remove_puncuations, text_to_sequence
from utils import load_json, write_json, parallel_run, remove_postfix, backup_file

# Patch 20180219: align by word not letter
def plain_text(text):
	return "".join(remove_puncuations(text.strip().lower()).split())

def plain_text_spaced(text):
	return " ".join(remove_puncuations(text.strip().lower()).split())

def add_punctuation(text):
	if text.endswith('다'):
		return text + "."
	else:
		return text

def similarity(text_a, text_b):
	text_a = plain_text(text_a)
	text_b = plain_text(text_b)

	score = SequenceMatcher(None, text_a, text_b).ratio()
	return score

def first_word_combined_words(text):
	# gives recognized first word, and recognized first+second word.
	words = text.split()
	if len(words) > 1:
		first_words = [words[0], words[0]+words[1]]
	else:
		first_words = [words[0]]
	return first_words

def first_word_combined_texts(text):
	# gives recognized first word, remainder and first+second word, remainder
	words = text.split()
	if len(words) > 1:
		if len(words) > 2:
			text2 = " ".join([words[0]+words[1]] + words[2:])
		else:
			text2 = words[0]+words[1]
		texts = [text, text2]
	else:
		texts = [text]
	return texts

def find_full_word(text,target_word, check_reverse=False):
	words_text=text.split()
	# first, find exact match
	try:
		exactloc=words_text.index(target_word)
		return target_word,exactloc
	# second, find not-so-exact-match
	except:
		for word in words_text:
			start_idx=word.find(target_word)
			if start_idx==-1: # not found
				if check_reverse and target_word.find(word) != -1: # check reverse
					return word, words_text.index(word)
				continue
			else:
				return word, words_text.index(word)
	return target_word,-1
			

def search_optimal(found_text, recognition_text, by_word=False):
	# 1. found_text is usually more accurate
	# 2. recognition_text can have more or less word
	# 3. (0219) recognition_test may have missed some heading syllable (e.g. for-> 'or', needles -> 'needle') (by_word switch)

	optimal = None
	found = False
	# some has bad spacing 
	recognition_text = recognition_text.strip() 
	found_text = found_text.strip()
	
	if plain_text(recognition_text) in plain_text(found_text):
		# BEST CASE : recognized text is in found_text
		
		# in case of missing heading syllable.
		if by_word:
			# find what the word is
			start_idx=plain_text_spaced(found_text).find(plain_text_spaced(recognition_text))
			firstwordloc=0
			endwordloc=-1
			if start_idx != -1: # if found
				words_plain_fromstart=plain_text_spaced(found_text)[start_idx:].split()
				words_plain_found=plain_text_spaced(found_text).split()
				# first find exact match of second word
				if len(words_plain_fromstart)>1:
					# find the location of second word, if it is available
					loc=words_plain_found.index(words_plain_fromstart[1])
					if loc==0:
						# second word cannot come first
						found = False
					elif words_plain_found[loc-1].find(words_plain_fromstart[0]) == -1:
						# first word(even broken) should be before the second word
						found = False
					else:
						found = True
						firstwordloc=loc-1
						
				if not found:
					firstword,firstwordloc=find_full_word(plain_text_spaced(found_text),words_plain_fromstart[0])
					found = True
					
				# find the location of last word
				recognition_last_word=plain_text_spaced(recognition_text).split()[-1]
				
				temp,endwordloc=find_full_word(plain_text_spaced(found_text),recognition_last_word)
				if endwordloc!=-1:
					optimal=" ".join(found_text.split()[firstwordloc:endwordloc+1])
				else:
					optimal=" ".join(found_text.split()[firstwordloc:])
				
				found = True
				if endwordloc < firstwordloc: found=False # if duplicate end detected
					
		if not found:
			# if such search failed, return to the recognized
			optimal = recognition_text
			found = True
		
	else:
		# if exact match is not found
		found = False
		for tmp_text in first_word_combined_texts(found_text):
			for recognition_first_word in first_word_combined_words(recognition_text):
				if recognition_first_word in tmp_text:
					start_idx = tmp_text.find(recognition_first_word)
					# in case of missing heading syllable
					if by_word:
						# 1. Find where is the first word.
						firstword,firstwordloc=find_full_word(plain_text_spaced(tmp_text),recognition_first_word.lower())
						# 2. reset the start_idx to follow real first word(from tmp_text, not recog.)
						start_idx=tmp_text.lower().find(firstword)
						found = True
						# Finally, check whether second word is also detected (if it exists)
						if len(plain_text_spaced(tmp_text)) > firstwordloc+1:
							secondword=plain_text_spaced(tmp_text)[firstwordloc+1]
							temp,idx=find_full_word(plain_text_spaced(recognition_text),secondword, check_reverse=True)
							# if second word not found in recognition (even in reverse)
							if idx == -1:
								found = False
					if tmp_text != found_text:
						found_text = found_text[max(0, start_idx-1):].strip()
					else:
						found_text = found_text[start_idx:].strip()
					
					break

			if found:
				break
				
		#found=False
		recognition_last_word = recognition_text.split()[-1]
		if recognition_last_word in found_text:
			end_idx = found_text.find(recognition_last_word)			
			punctuation = ""

			if len(found_text) > end_idx + len(recognition_last_word):
				punctuation = found_text[end_idx + len(recognition_last_word)]
				if punctuation not in string.punctuation:
					punctuation = ""

			found_text = found_text[:end_idx] + recognition_last_word + punctuation
			# in case of missing tailing syllable - find full last word and replace it.
			'''
			if by_word:
				found_text_last_word, temp=find_full_word(plain_text_spaced(found_text),recognition_last_word.lower())
				found_text= found_text[:end_idx] + remove_puncuations(found_text_last_word) + punctuation
			found = True
			'''

		if found or by_word == False:
			optimal = found_text
		else:
			# pros - remove unsaid heading/trailing elements
			# cons - cannot align misrecognized head/tail
			optimal = None 

	return optimal


def align_text_fn(
		item, score_threshold, debug=False):

	audio_path, recognition_text = item

	audio_dir = os.path.dirname(audio_path)
	base_dir = os.path.dirname(audio_dir)

	news_path = remove_postfix(audio_path.replace("audio", "assets"))
	news_path = os.path.splitext(news_path)[0] + ".txt"

	strip_fn = lambda line: line.strip().replace('"', '').replace("'", "")
	candidates = [strip_fn(line) for line in open(news_path, encoding='cp949').readlines()]

	scores = { candidate: similarity(candidate, recognition_text) \
					for candidate in candidates}
	sorted_scores = sorted(scores.items(), key=operator.itemgetter(1))[::-1]

	first, second = sorted_scores[0], sorted_scores[1]

	if first[1] > second[1] and first[1] >= score_threshold:
		found_text, score = first
		aligned_text = search_optimal(found_text, recognition_text, by_word=True)

		if debug: 
			print("	  ", audio_path)
			print("	  ", recognition_text)
			print("=> ", found_text)
			print("==>", aligned_text)
			print("="*30)

		if aligned_text is not None:
			result = { audio_path: add_punctuation(aligned_text) }
		elif abs(len(text_to_sequence(found_text)) - len(text_to_sequence(recognition_text))) > 10:
			result = {}
		else:
			result = { audio_path: [add_punctuation(found_text), recognition_text] }
	else:
		result = {}

	if len(result) == 0:
		result = { audio_path: [recognition_text] }

	return result

def align_text_batch(config):
	align_text = partial(align_text_fn,
			score_threshold=config.score_threshold)

	results = {}
	data = load_json(config.recognition_path, encoding=config.recognition_encoding)

	items = parallel_run(
			align_text, data.items(),
			desc="align_text_batch", parallel=True)

	for item in items:
		results.update(item)

	found_count = sum([type(value) == str for value in results.values()])
	print(" [*] # found: {:.5f}% ({}/{})".format(
			len(results)/len(data), len(results), len(data)))
	print(" [*] # exact match: {:.5f}% ({}/{})".format(
			found_count/len(items), found_count, len(items)))

	return results

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--recognition_path', required=True)
	parser.add_argument('--alignment_filename', default="alignment.json")
	parser.add_argument('--score_threshold', default=0.4, type=float)
	parser.add_argument('--recognition_encoding', default='949')
	config, unparsed = parser.parse_known_args()

	results = align_text_batch(config)

	base_dir = os.path.dirname(config.recognition_path)
	alignment_path = \
			os.path.join(base_dir, config.alignment_filename)

	if os.path.exists(alignment_path):
		backup_file(alignment_path)

	write_json(alignment_path, results)
	duration = get_durations(results.keys(), print_detail=False)
