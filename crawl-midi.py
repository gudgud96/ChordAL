'''
Author:     Tan Hao Hao
Project:    deeppop
Purpose:    Scrape midi files from midiworld.com
'''
import urllib.request
from bs4 import BeautifulSoup
import re
from tqdm import tqdm

for i in tqdm(range(35, 38)):
    url = 'http://www.midiworld.com/search/' + str(i) + '/?q=pop'
    response = urllib.request.urlopen(url)
    data = response.read()      # a `bytes` object
    text = data.decode('utf-8')

    soup = BeautifulSoup(text, 'html.parser')
    li_elems = soup.find_all('li')
    songnames = []
    links = []

    for elem in li_elems:
        elem = str(elem).replace('\n', '')
        if '>download<' in elem:
            songname = re.search('<li>.*\s-', elem).group().replace('<li>', '').replace('-', '').rstrip()
            songnames.append(songname)
            link = re.search('<a href=".* target', elem).group().replace('<a href="', '').replace('" target', '').rstrip()
            links.append(link)

    for i in range(len(links)):
        urllib.request.urlretrieve(links[i], 'Pop Midis/' + songnames[i] + '.mid')


