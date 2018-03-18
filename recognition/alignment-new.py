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
# Patch 20180313: Very rigorous search (all case search)
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
	# try all possible alignment of found_text and find best alignment
	
	optimal = None
	best_start=0
	best_end=0
	best_score=0.0
	
	# some has bad spacing 
	recognition_text = recognition_text.strip() 
	found_text = found_text.strip()
	# consider spacing error
	compare_recognition_text = ''.join(plain_text_spaced(recognition_text).split())
	found_text_split = plain_text_spaced(found_text).split()
	
	for start in range(len(found_text_split)):
		for end in range(start+1,len(found_text_split)+1):
			compare_found_text = ''.join(found_text_split[start:end])
			score = similarity(compare_found_text,compare_recognition_text)
			if score > best_score:
				best_score = score
				best_start = start
				best_end = end
	
	found_text_split = found_text.split()
	optimal = ' '.join(found_text_split[best_start:best_end])
	
	return optimal, best_score


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
	
	'''
	if first[1] > second[1] and first[1] >= score_threshold:
		found_text, score = first
		aligned_text, align_score = search_optimal(found_text, recognition_text, by_word=True)
	'''
	best_aligned_text = ''
	found_text=''
	best_simscore = 0.0
	for candidate, simscore in sorted_scores:
		aligned_text, simscore = search_optimal(candidate, recognition_text)
		if simscore > best_simscore:
			best_aligned_text = aligned_text
			best_simscore = simscore
			found_text=candidate
		if simscore == 1.0:
			break
	
	aligned_text = best_aligned_text
	if len(best_aligned_text) != 0 and best_simscore >= score_threshold:
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
