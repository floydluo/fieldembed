./jwe/src/jwe -train ./corpus/WikiTotal/WikiTotal7k_v2.txt \
			  -output-word embeddings/jwe/word_vec \
			  -output-char embeddings/jwe/char_vec \
			  -output-comp embeddings/jwe/comp_vec \
			  -size 200 \
			  -window 5 \
			  -negative 10 \
			  -sample 1e-5 \
			  -iter 1 \
			  -threads 4\
			  -min-count 10 \
			  -alpha 0.025 \
			  -comp ./jwe/subcharacter/comp.txt \
			  -char2comp ./jwe/subcharacter/char2comp.txt \
			  -join-type 1 \
			  -pos-type 1 \
			  -average-sum 1


# ./word2vec/word2vec -train ./corpus/WikiTotal/WikiTotal7k_v2.txt \
# 			        -output embeddings/word2vec/word_vec \
# 			        -cbow 1 \
# 			        -size 200 \
# 			        -window 5 \
# 			        -negative 10\
# 			        -sample 1e-5 \
# 			        -iter 1\
# 			        -threads 4\
# 			        -min-count 10 \
# 			  		-alpha 0.025
			        
