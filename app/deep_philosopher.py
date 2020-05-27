# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# %%
from fastai.text import *
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

book = epub.read_epub('data/descartes1641.epub')


# %%
all_par = []

for doc in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
    if doc.is_chapter():
        chap_raw = doc.get_content()
        soup = BeautifulSoup(chap_raw)
        for par in soup.find_all('p', text=False, recursive=True):
            # filter out paragraphs with less then 50 characters
            if len(par.text) > 50:     
                try:
                    all_par = all_par + [par.text]
                except:
                    print(f'Not parsable: {par}')
    
df_texts = pd.DataFrame(all_par[1::], columns=['text'])
df_texts.shape[0]
df_texts.head()

# %%
df_texts['text'][1]


# %%
# Language model
bs = 48
# stor_path = Path('/storage/deep-philosopher/')
stor_path = Path('/Users/jobvisser/repos/deep-philosopher/app')

data_lm = (TextList.from_df(df_texts, stor_path, cols=['text'])
            .random_split_by_pct(.1)
            .label_for_lm()
            .databunch(bs=bs))

data_lm.save(stor_path/'static/data_lm.pkl')

#%%
data_lm = load_data(stor_path/'static', 'data_lm.pkl', bs=bs)

# %%
data_lm.show_batch(rows=10)

# %%
# view the most common dictionary terms
data_lm.vocab.itos[:15]

# %%
# Train Language Model
# drop_mult is a parameter that controls the % of drop-out used
learn = language_model_learner(data_lm, AWD_LSTM, 
            drop_mult=0.7)

# %%
# use learning rate finder to identify a good learning rate to use
learn.lr_find()

# %%
learn.recorder.plot(skip_end=15)

# %%
# as a rule of thumb, review the plot above and choose the learning rate with the steepest slope to fit the model
learn.fit_one_cycle(10, 1e-2, moms=(0.8,0.7))

# %%
learn.save(stor_path/'models/deep_philosopher_head')

# %%
learn.load(stor_path/'models/deep_philosopher_head');

# %%
learn.fit_one_cycle(5, 1e-3, moms=(0.8,0.7))

# %%
learn.save(stor_path/'models/deep_philosopher_fine_tuned')

# %%
learn.load(stor_path/'models/deep_philosopher_fine_tuned')

# %%
# adjust below parameters to test inference (generated texts)

TEXT = "Life will"
N_WORDS = 24
N_SENTENCES = 2

# %%
print("\n".join(learn.predict(TEXT, N_WORDS) for _ in range(N_SENTENCES)))

# %%
