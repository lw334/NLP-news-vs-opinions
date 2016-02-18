__author__ = 'weiwei'
import urllib.request, time, re, random, hashlib
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk import word_tokenize
from nltk import pos_tag
from sklearn.svm import LinearSVC as LSVC
import settings

OPINION = 1
NEWS = 0
TRAIN = 1
TEST = 0
last_fetched_at = None

# Compassionate Caching inspired by
# http://lethain.com/an-introduction-to-compassionate-screenscraping/
def fetch(url):
    """Load the url compassionately."""

    global last_fetched_at

    url_hash = hashlib.sha1(url.encode()).hexdigest()
    filename = 'cache-file-{}'.format(url_hash)
    print(url_hash)
    try:
        with open(filename, 'r') as f:
            result = f.read()
            if len(result) > 0:
                print("Retrieving from cache:", url)
                return result
    except:
        pass

    print("Loading:", url)
    wait_interval = random.randint(3000,10000)
    if last_fetched_at is not None:
        now = time.time()
        elapsed = now - last_fetched_at
        if elapsed < wait_interval:
            time.sleep((wait_interval - elapsed)/1000)

    user_agent = 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)'
    headers = { 'User-Agent' : user_agent }
    req = urllib.request.Request(url, headers = headers)
    last_fetched_at = time.time()
    with urllib.request.urlopen(req) as response:
        result = str(response.read())
        with open(filename, 'w') as f:
            f.write(result)
        return result

# Get a set of links for news articles
def get_news_links(link):
    articles = fetch(link)
    soup = BeautifulSoup(articles, 'html.parser')
    links = set([])
    pattern = re.compile(r"^http://www\.factcheck\.org/\d{4}/\d{2}/")
    for a in soup.find_all("a"):
        sub_link = a.get("href")
        if sub_link != None and pattern.match(sub_link):
            links.add(sub_link)
    return links

def news_scraper():
    return news_set

def opinion_scraper():
    return opinion_set


# In[7]:

opinion_set = {'http://www.usatoday.com/story/opinion/2013/09/16/syria-irs-lerner-column/2816277/',
 'http://www.usatoday.com/story/opinion/2014/05/12/president-obama-irs-scandal-watergate-column/8968317/',
 'http://www.usatoday.com/story/opinion/2014/08/28/russia-ukraine-nato-vladimir-putin-president-obama-editorials-debates/14766425/',
 'http://www.usatoday.com/story/opinion/2015/05/25/caption-contest-youtoon/1568271/',
 'http://www.usatoday.com/story/opinion/2015/07/23/highway-funding-oil-gasoline-tax-fix-congress-editorials-debates/30579385/',
 'http://www.usatoday.com/story/opinion/2015/12/21/cdc-opioids-heroin-overdoses-doctors-editorials-debates/77708774/',
 'http://www.usatoday.com/story/opinion/2016/02/01/mia-love-single-subject-rule-constitutional-amendment--reynolds-column/79605158/',
 'http://www.usatoday.com/story/opinion/2016/02/01/super-bowl-football-brain-damage-immoral-watch-column/79654086/',
 'http://www.usatoday.com/story/opinion/2016/02/04/obama--wrong-solitary-confinement-column/79649416/',
 'http://www.usatoday.com/story/opinion/2016/02/04/trump-sanders-mccain-new-hampshire-mavericks-column/79832920/'}

news_set = {'http://www.factcheck.org/2011/11/we-repeat-still-a-christmas-tree/',
            'http://www.factcheck.org/2012/04/death-panels-redux/',
            'http://www.factcheck.org/2012/03/alaskan-island-giveaway/',
            'http://www.factcheck.org/2012/02/did-obama-approve-bridge-work-for-chinese-firms/',
            'http://www.factcheck.org/2012/02/dueling-debt-deceptions/',
            'http://www.factcheck.org/2012/01/neurological-death-panels/',
            'http://www.factcheck.org/2011/12/ron-pelosis-connection-to-tonopah-solar-energy/',
            'http://www.factcheck.org/2011/12/the-gingrich-divorce-myth/',
            'http://www.factcheck.org/2011/12/ron-pelosis-connection-to-tonopah-solar-energy/',
            'http://www.factcheck.org/2014/10/midterm-medicare-mudslinging/'}

# ### 2. Preprocessing & Feature Generation


STOP_WORDS = stopwords.words('english')
STOP_PHRASES = ["Ask FactCheck", "FULL QUESTION", "FULL ANSWER", '© Copyright 2016 FactCheck.org', 'A Project of the Annenberg Public Policy Center']
def get_words(article_html, is_opinion):
    """Return list of representative words from an article. """
    bag_of_words = []
    raw = []
    if not is_opinion:
        additional = re.search(r'<span style="color:.{,20}">(<strong>)?Sources(</strong>)?</span>', article_html)
        if additional:
            article_html = article_html[:additional.start()]
    soupify_article = BeautifulSoup(article_html, 'html.parser')
    paragraphs = soupify_article.find_all('p',attrs={'class':None})
    for p in paragraphs:
        if p.parent.name != 'a' and p.text not in STOP_PHRASES:
            p_text = p.text.lower().replace('usa today', ' ').replace('q: ', ' ').replace('a: ', ' ').replace('getelementbyid', ' ').replace('eet', ' ')
            raw += word_tokenize(p_text)
    for word in raw:
        if '\\xc2\\xa0' in word:
            tmp = word.split('\\xc2\\xa0')
        else:
            tmp = [word]
        tmp = [re.sub(r"\\x..", "", w).replace("\\", "") for w in tmp]
        for w in tmp:
            bag_of_words += re.sub(r"[^a-zA-Z]", " ", w).split()
    bag_of_words = [w.lower() for w in bag_of_words if w.isalpha() and w not in STOP_WORDS]
    return bag_of_words

def build_corpus(opinion_set, news_set):
    opinion = [(get_words(fetch(link), OPINION), OPINION) for link in opinion_set]
    news = [(get_words(fetch(link), NEWS), NEWS) for link in news_set]
    corpus = news + opinion
    random.shuffle(corpus)
    return corpus

def vectorize(vectorizer, list_of_texts, is_train):
    """Return feature vectors for each entity given list of texts."""
    if is_train:
        compressed_vectors = vectorizer.fit_transform(list_of_texts)
    else:
        compressed_vectors = vectorizer.transform(list_of_texts)
    return compressed_vectors.toarray()


# ### 3. SVC classifier

def prepare_clf():
    news_set = news_scraper()
    opinion_set = opinion_scraper()
    corpus = build_corpus(opinion_set, news_set)
    train(corpus)

def train(data):
    vectorizer = TfidfVectorizer(analyzer = "word", stop_words = "english", ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
    X_train = vectorize(vectorizer, [' '.join(article) + ' ' + ' '.join(t for w,t in nltk.pos_tag(article)) for article, tag in data], TRAIN)
    y_train = [tag for article, tag in data]
    settings.clf = LSVC().fit(X_train, y_train)
    settings.vectorizer = vectorizer

def predict_sample(article, vectorizer, clf):
    sample = vectorize(vectorizer, [' '.join(article) + ' ' + ' '.join(t for w,t in nltk.pos_tag(article))], TEST)
    y_pred = clf.predict(sample)
    return y_pred

