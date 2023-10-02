shuf -n 10000 books_large_p1.txt > book_large_p1.txt.sample_10k
perl detokenizer.perl -u < book_large_p1.txt.sample_10k > book_large_p1.txt.sample_10k.detok