import os 
import argparse


join = lambda x,y: os.path.join(x,y)
isdir = lambda x: os.path.isdir(x)
isfile = lambda y: os.path.isfile(y)
exists = lambda z : os.path.exists(z)
header = '__label__'

"""

  USAGE :  python fasttext_input.py  --directory <directory to contatining documents> --truthfile <file with labels> --outfile <name of out file>

"""

def get_truth_dict(truthfile):
	truth = {}
	with open(truthfile) as tf:
		for line in tf:
			pieces = line.split()
			subject = pieces[0].split('_')[-1]
			truth[subject]=pieces[-1]
	return truth

def get_document_content(document):
	with open(document,'r',errors='ignore') as doc:
		return doc.read().replace('\n', '')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='USAGE :python fasttext_input.py  --directory <directory to contatining documents> --truthfile <file with labels> --outfile <name of out file>')
	parser.add_argument('-d','--directory',help='',required=True)
	parser.add_argument('-t','--truthfile',help='',required=True)
	parser.add_argument('-o','--outfile',help='',required=True)
	args= vars(parser.parse_args())

	documents = ( d for d in os.listdir(args['directory']) if isfile(join(args['directory'],d)))
	truth = get_truth_dict(args['truthfile'])

	with open(args['outfile'],'w',encoding='utf-8',errors='ignore') as out:
		for doc in documents:
			to_write = header
			subject = doc.split('.')[0]
			if 'train' in subject:
				subject = subject.split('_')[-2]
			to_write+=truth[subject]+' '
			to_write+=get_document_content(join(args['directory'],doc))
			out.write(to_write)
			out.write('\n')

