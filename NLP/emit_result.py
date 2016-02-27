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

def news_scraper(n=10):
    links = set([])
    for i in range(1, n+1):
        link = "http://www.factcheck.org/askfactcheck/page/{}/".format(i)
        links = links.union(get_news_links(link))
    return links

OPINION_SET = {'http://www.usatoday.com/story/opinion/2013/09/16/syria-irs-lerner-column/2816277/',
 'http://www.usatoday.com/story/opinion/2013/09/25/grocery-store-detroit-irs-column/2868797/',
 'http://www.usatoday.com/story/opinion/2014/05/12/president-obama-irs-scandal-watergate-column/8968317/',
 'http://www.usatoday.com/story/opinion/2014/08/28/russia-ukraine-nato-vladimir-putin-president-obama-editorials-debates/14766425/',
 'http://www.usatoday.com/story/opinion/2015/05/25/caption-contest-youtoon/1568271/',
 'http://www.usatoday.com/story/opinion/2015/07/23/highway-funding-oil-gasoline-tax-fix-congress-editorials-debates/30579385/',
 'http://www.usatoday.com/story/opinion/2015/12/21/cdc-opioids-heroin-overdoses-doctors-editorials-debates/77708774/',
 'http://www.usatoday.com/story/opinion/2016/02/01/mia-love-single-subject-rule-constitutional-amendment--reynolds-column/79605158/',
 'http://www.usatoday.com/story/opinion/2016/02/01/super-bowl-football-brain-damage-immoral-watch-column/79654086/',
 'http://www.usatoday.com/story/opinion/2016/02/04/obama--wrong-solitary-confinement-column/79649416/',
 'http://www.usatoday.com/story/opinion/2016/02/04/trump-sanders-mccain-new-hampshire-mavericks-column/79832920/',
 'http://www.usatoday.com/story/opinion/2016/02/07/journalists-jail-murder-censorship-turkey-editorials-debates/79844586/',
 'http://www.usatoday.com/story/opinion/2016/02/07/new-hampshire-primary-100-years-old-rebel-role-dante-scala-column/79967400/',
 'http://www.usatoday.com/story/opinion/2016/02/07/police-use-of-lethal-force-tellusatoday-your-say/79978876/',
 'http://www.usatoday.com/story/opinion/2016/02/07/turkish-ambassador-journalists-turkey-editorials-debates/79845450/',
 'http://www.usatoday.com/story/opinion/2016/02/07/voter-anger-elections-super-bowl-second-look/79967622/',
 'http://www.usatoday.com/story/opinion/2016/02/08/bill-de-blasio-chirlane-mccray-opioid-crisis-treatment-naloxone-overdoses-column/79972594/',
 'http://www.usatoday.com/story/opinion/2016/02/08/cal-thomas-elections-2016-god-religion-politics-evangelical-voters-column/79943324/',
 'http://www.usatoday.com/story/opinion/2016/02/08/federal-deficit-our-view-editorials-debates/80024164/',
 'http://www.usatoday.com/story/opinion/2016/02/08/federal-deficits-economy-governemtn-spending-editorials-debates/80032380/',
 'http://www.usatoday.com/story/opinion/2016/02/08/irs-tea-party-targeting-lois-lerner-corruption--obama-glenn-reynolds-column/79967098/',
 'http://www.usatoday.com/story/opinion/2016/02/08/martin-shkreli-drug-prices-your-say/80026236/',
 'http://www.usatoday.com/story/opinion/2016/02/08/primary-voting-presidential-election-tellusatoday-your-say/80026468/',
 'http://www.usatoday.com/story/opinion/2016/02/09/bernie-sanders-hillary-clinton-new-hampshire-column/80094342/',
 'http://www.usatoday.com/story/opinion/2016/02/09/beyonce-ads-super-bowl-colbert-corden-meyers-conan-jessica-williams/80052554/',
 'http://www.usatoday.com/story/opinion/2016/02/09/military-medical-battlefield-training-live-tissue-training-animal-rights-column/80018116/',
 'http://www.usatoday.com/story/opinion/2016/02/09/new-hampshire-primary-donald-trump-bernie-sanders-editorials-debates/80091284/',
 'http://www.usatoday.com/story/opinion/2016/02/09/obama-administration-least-transparent-epa-state-doj-clinton-benghazi-column/80050428/',
 'http://www.usatoday.com/story/opinion/2016/02/09/our-votes-matter-voter-id-citizens-united-voting-rights-act-democracy-awakens-column/80068028/',
 'http://www.usatoday.com/story/opinion/2016/02/09/solitary-confinement-federal-prisons-tellusatoday-your-say/80086320/',
 'http://www.usatoday.com/story/opinion/2016/02/09/super-bowl-50-your-say/80086738/',
 'http://www.usatoday.com/story/opinion/2016/02/09/trump-sanders-wins-new-hampshire-economic-anxiety-column/80088548/',
 'http://www.usatoday.com/story/opinion/2016/02/10/anthem-cruise-ship-storm-your-say/80202290/',
 'http://www.usatoday.com/story/opinion/2016/02/10/colbert-noah-fallon-kimmel-corden-sanders-trump-punchlines-new-hampshire/80179418/',
 'http://www.usatoday.com/story/opinion/2016/02/10/exonerations-dna-convicted-forensic-criminal-justice-column/80056392/',
 'http://www.usatoday.com/story/opinion/2016/02/10/hillary-clinton-women-voters-millennials-new-hampshire-column/80190950/',
 'http://www.usatoday.com/story/opinion/2016/02/10/hillary-clintons-woman-problem-column/80175130/',
 'http://www.usatoday.com/story/opinion/2016/02/10/marco-rubio-hip-hop-ben-carson-trump-bush-young-minority-voters-column/76387044/',
 'http://www.usatoday.com/story/opinion/2016/02/10/new-hampshire-primary-donald-trump-bernie-sanders-tellusatoday-your-say/80202062/',
 'http://www.usatoday.com/story/opinion/2016/02/10/oil-prices-gasoline-revenue-american-petroleum-institute-editorials-debates/80193760/',
 'http://www.usatoday.com/story/opinion/2016/02/10/oil-tax-10-barrel-infrastructure-president-obama-climate-change-editorials-debates/80056688/',
 'http://www.usatoday.com/story/opinion/2016/02/10/why-supreme-court-put-new-climate-rules-hold-column/80169792/',
 'http://www.usatoday.com/story/opinion/2016/02/11/federal-budget-obama-deficits-debt-tellusatoday-your-say/80253310/',
 'http://www.usatoday.com/story/opinion/2016/02/11/glenn-reynolds-socialism-bernie-sanders-young-millennial-voters-column/80169668/',
 'http://www.usatoday.com/story/opinion/2016/02/11/hillary-clinton-bernie-sanders-wall-street-lanny-davis-editorials-debates/80253414/',
 'http://www.usatoday.com/story/opinion/2016/02/11/hillary-clinton-speeches-goldman-sachs-wall-street-speaking-fees-editorials-debates/80233010/',
 'http://www.usatoday.com/story/opinion/2016/02/11/obama-budget-children-summer-food-hope-change-david-cay-johnston/80199860/',
 'http://www.usatoday.com/story/opinion/2016/02/11/wesley-clark-russia-assadsyria-obama-conflict-column/80228140/',
 'http://www.usatoday.com/story/opinion/2016/02/12/ligo-discovery-impossible-without-public-funding-gravitational-waves-column/80253446/',
 'http://www.usatoday.com/story/opinion/2016/02/12/lindberg-draft-conscription-women-all-volunteer-force-courage-virtue-column/80169484/',
 'http://www.usatoday.com/story/opinion/2016/02/12/top-threat-kurds-economy-not-isil-column/80228512/',
 'http://www.usatoday.com/story/opinion/2016/02/12/valentines-day-jimmy-kimmel-james-corden-punchlines-funny/80289898/',
 'http://www.usatoday.com/story/opinion/2016/02/13/scalia-death-appreciation-politics-nomination-glenn-reynolds-column/80350008/',
 'http://www.usatoday.com/story/opinion/2016/02/13/scalia-text-legacy-clerk-steven-calabresi-column/80349810/',
 'http://www.usatoday.com/story/opinion/2016/02/13/valentines-day-romance-marraige-flowers-fracking-column/80234586/',
 'http://www.usatoday.com/story/opinion/2016/02/14/antonin-scalia-2016-presidential-election-voters-editorials-debates/80382050/',
 'http://www.usatoday.com/story/opinion/2016/02/14/antonin-scalia-death-supreme-court-nomination-senate-obama-gonzales-column/80378246/',
 'http://www.usatoday.com/story/opinion/2016/02/14/bernie-sanders-henry-kissinger-richard-nixon-democratic-debate-column/80372646/',
 'http://www.usatoday.com/story/opinion/2016/02/14/justice-antonin-scalia-president-obama-mitch-mcconnell-editorials-debates/80375514/',
 'http://www.usatoday.com/story/opinion/2016/02/14/martin-shkreli-cam-newton-second-look-your-say/80383482/',
 'http://www.usatoday.com/story/opinion/2016/02/14/oil-tax-transportation-president-obama-your-say/80383560/',
 'http://www.usatoday.com/story/opinion/2016/02/14/religion-politics-gender-tellusatoday-your-say/80383622/',
 'http://www.usatoday.com/story/opinion/2016/02/14/scalia-defining-moment-minority-rights-stephen-henderson/80372366/',
 'http://www.usatoday.com/story/opinion/2016/02/14/why-i-wrote-play-antonin-scalia-originalist-john-strand/80374808/',
 'http://www.usatoday.com/story/opinion/2016/02/15/american-kennel-club-westminster-kennel-club-dog-show-editorials-debates/80401688/',
 'http://www.usatoday.com/story/opinion/2016/02/15/antonin-scalia-supreme-court-recess-appointment-nomination-politics-obama-column/80379796/',
 'http://www.usatoday.com/story/opinion/2016/02/15/dogs-breeding-westminster-kennel-american-kennel-club-editorials-debates/80373002/',
 'http://www.usatoday.com/story/opinion/2016/02/15/donald-trump-torture-enhanced-interrogation-techniques-editorials-debates/80418458/',
 'http://www.usatoday.com/story/opinion/2016/02/15/donald-trump-waterboarding-torture-editorials-debates/80258136/',
 'http://www.usatoday.com/story/opinion/2016/02/15/gop-supreme-court-scalia-obama-nominee-tellusatoday-your-say/80425956/',
 'http://www.usatoday.com/story/opinion/2016/02/15/hillary-clinton-feminism-sexism-bernie-bros-democratic-primary-2016-column/80374526/',
 'http://www.usatoday.com/story/opinion/2016/02/15/jim-wallis-getting-personal-racism-black-lives-matter/79977654/',
 'http://www.usatoday.com/story/opinion/2016/02/15/john-oliver-colin-jost-michael-che-punchlines-democracy-voting/80405220/',
 'http://www.usatoday.com/story/opinion/2016/02/15/patrick-leahy-antonin-scalia-death-supreme-court-nomination-confirmation-column/80415542/',
 'http://www.usatoday.com/story/opinion/2016/02/15/supreme-court-fight-assures-ugly-end-obama-era-david-corn-antonin-scalia-column/80374474/',
 'http://www.usatoday.com/story/opinion/2016/02/15/trump-has-no-idea-how-to-be-president-stephen-hess/80401590/',
 'http://www.usatoday.com/story/opinion/2016/02/15/wealthy-donors-citizens-united-politics-your-say/80425588/',
 'http://www.usatoday.com/story/opinion/2016/02/16/doj-ferguson-lawsuit-police-tellusatoday-your-say/80479008/',
 'http://www.usatoday.com/story/opinion/2016/02/16/evangelicals-south-carolina-republican-primary-column/80414280/',
 'http://www.usatoday.com/story/opinion/2016/02/16/hillary-clinton-bernie-sanders-nevada-caucuses-jon-ralston/80450100/',
 'http://www.usatoday.com/story/opinion/2016/02/16/kirsten-powers-bernie-sanders-hillary-clinton-democratic-primary-2016-column/80407150/',
 'http://www.usatoday.com/story/opinion/2016/02/16/libya-islamic-state-isil-oil-terrorism-obama-daesh-column/80018234/',
 'http://www.usatoday.com/story/opinion/2016/02/16/mlb-lifetime-ban-jenrry-mejia-peds-your-say/80478800/',
 'http://www.usatoday.com/story/opinion/2016/02/16/scalia-supreme-court-alexander-hamilton-musical-nomination-senate-obama-column/80465232/',
 'http://www.usatoday.com/story/opinion/2016/02/16/scalia-supreme-court-obama-gop-punchlines-bee-meyers/80451096/',
 'http://www.usatoday.com/story/opinion/2016/02/17/best-supreme-court-nominee-depends-jonathan-turley/80516622/',
 'http://www.usatoday.com/story/opinion/2016/02/17/cable-tv-set-top-box-fcc-tom-wheeler-editorials-debates/80474618/',
 'http://www.usatoday.com/story/opinion/2016/02/17/irs-civil-asset-forfeiture-ken-quran-randy-sowers-institute-justice-column/80499524/',
 'http://www.usatoday.com/story/opinion/2016/02/17/kanye-swift-fallon-colbert-corden-grammys-punchlines/80503382/',
 'http://www.usatoday.com/story/opinion/2016/02/17/lawrence-lessig-scalia-set-principled-example/80448256/',
 'http://www.usatoday.com/story/opinion/2016/02/17/randy-barnett-antonin-scalia-new-originalism-heller-second-amendment-column/80450446/',
 'http://www.usatoday.com/story/opinion/2016/02/17/tevision-cable-fcc-tom-wheeler-google-editorials-debates/80519326/',
 'http://www.usatoday.com/story/opinion/2016/02/17/wwjd-vote-for-bernie-sanders-column/80426466/',
 'http://www.usatoday.com/story/opinion/2016/02/17/yoweri-mouseveni-uganda-african-leaders-term-limits-obama-column/79651582/',
 'http://www.usatoday.com/story/opinion/columnists/stephen-henderson/2016/02/13/moments-defined-scalia-and-should-define-legacy/80355476/',
 'http://www.usatoday.com/story/opinion/columnists/stephen-henderson/2016/02/16/alexander-hamilton-and-looming-high-court-battle/80459026/',
 'http://www.usatoday.com/story/opinion/voices/2016/02/08/voices-rise-and-fall-rand-paul/79875100/',
 'http://www.usatoday.com/story/opinion/voices/2016/02/09/voices-mexico-legalize-marijuana/79781382/',
 'http://www.usatoday.com/story/opinion/voices/2016/02/10/voices-staying-safe-dangerous-venues/80170178/',
 'http://www.usatoday.com/story/opinion/voices/2016/02/15/voices-gomez-honduras-violence-central-america-unaccompanied-minors-immigration/80212272/'}


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

def get_words_from_string(article):
    return [w.lower() for w in nltk.word_tokenize(article) if w.isalpha() and (w not in STOP_WORDS or w.lower() not in STOP_WORDS)]

def prepare_clf():
    news_set = news_scraper()
    opinion_set = OPINION_SET #opinion_scraper()
    corpus = build_corpus(opinion_set, news_set)
    train(corpus)

def train(data):
    vectorizer = TfidfVectorizer(analyzer = "word", stop_words = "english", ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
    X_train = vectorize(vectorizer, [' '.join(article) + ' ' + ' '.join(t for w,t in nltk.pos_tag(article)) for article, tag in data], TRAIN)
    y_train = [tag for article, tag in data]
    settings.clf = LSVC().fit(X_train, y_train)
    settings.vectorizer = vectorizer

def predict_sample(article, vectorizer, clf):
    article = get_words_from_string(article)
    sample = vectorize(vectorizer, [' '.join(article) + ' ' + ' '.join(t for w,t in nltk.pos_tag(article))], TEST)
    y_pred = clf.predict(sample)
    return y_pred

