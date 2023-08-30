#%%
import numpy as np

def min_edit_distance(source, target):
    n = len(source)
    m = len(target)

    D = np.zeros((n+1,m+1))

    for i in range(n+1):
        D[i,0] = i 

    for i in range(m+1):
        D[0,i] = i
        
    # Recursion
    for i in range(1,n+1):
        for j in range(1,m+1):
            D[i,j] = min(D[i-1,j] + 1, 
                        D[i,j-1] + 1, 
                        D[i-1,j-1] + 2 if source[i-1] != target[j-1] else D[i-1,j-1] + 0 if source[i-1] == target[j-1] else np.nan)
    return D[n,m]

min_edit_distance("intention", "execution")

#%%

old = ['bandelierskop',
'bathlaros',
'bavianskloof',
'belabela',
'bisho',
'bitye',
'bizana',
'boksburg-north',
'bolo',
'bolotwa',
'boshofffffff',
'bronkhorspruit',
'calcuta',
'capetown',
'carltonville',
'dealsville',
'engcobo',
'esikhawini',
'hammersdale',
'heidelberg',
'henneman',
'hilton',
'idutywa',
'johannesburg',
'kentani',
'kenton-on-sea',
'kidd sbeach',
'kingshakainternationalairport',
'kubusiedrift',
'kwa-thema',
'leeu-gamka',
'lethlabile',
'lowscreek',
'lyttleton',
'mafikeng',
'mahlabatini',
'makgobistat',
'mandini',
'mapumulo',
'mayville',
'middelburg',
'mkuze',
'mothutlong',
'mpumalangastation',
'muldersdrif',
'nqamakwe',
'nqutu',
'ntabamhlope',
'ortambointernationalairport',
'pholo',
'phuthaditjaba',
'pilgrimsrest',
'portelizabeth',
'rankinspass',
'richmond',
'riebeeckwest',
'schweizerreneke',
'senwabarana',
'tabankulu',
'thabanchu',
'tinafalls',
'ulundistation',
'umzimkulu']

new = ['alexandra',
'algoapark',
'athlone',
'atlantis',
'balfourtvl',
'bandelierkop',
'batlharos',
'baviaanskloof',
'bela-bela',
'belhar',
'bellville',
'bellvillesouth',
'bethelsdorp',
'bhisho',
'bholo',
'bholothwa',
'bishoplavis',
'bityi',
'bohlokong',
'boksburgnorth',
'booysens',
'boshof',
'bothasig',
'botshabelo',
'brackenfell',
'bramley',
'brixton',
'bronkhorstspruit',
'calcutta',
'campsbay',
'capetowncentral',
'carletonville',
'centane',
'claremont',
'cleveland',
'dealesville',
'delft',
'despatch',
'diepkloof',
'dieprivier',
'diepsloot',
'dobsonville',
'doornkop',
'douglasdale',
'durbanville',
'dutywa',
'eldoradopark',
'elsiesriver',
'ennerdale',
'esikhaleni',
'fairland',
'fishhoek',
'florida',
'gelvandale',
'goodwood',
'grassypark',
'gugulethu',
'hammarsdale',
'harare',
'heidelberg(c)',
'heidelberg(gp)',
'hennenman',
'hillbrow',
'hilton-kzn',
'honeydew',
'houtbay',
'humewood',
'ikamvelihle',
'intairpkingshaka',
'ivorypark',
'jabulani',
'jeppe',
'jhbcentral',
'kabegapark',
'kensington',
'kentononsea',
'khayelitsha',
'khubusidrift',
'kiddsbeach',
'kirstenhof',
'kleinvlei',
'kliptown',
'kraaifontein',
'kuilsrivier',
'kwadwesi',
'kwanobuhle',
'kwathema',
'kwazakele',
'langa',
'langlaagte',
'lansdowne',
'leeugamka',
'lenasia',
'lenasiasouth',
'lentegeur',
'letlhabile',
'linden',
'lingelethu-west',
'low screek',
'lwandle',
'lyttelton',
'mabeskraal',
'macassar',
'mahikeng',
'mahlabathini',
'maitland',
'makgobistad',
'mandeni',
'manenberg',
'maphumulo',
'mayville-kzn',
'mbizana',
'meadowlands',
'melkbosstrand',
'mfuleni',
'middelburg(ec)',
'middelburgmpu',
'midrand',
'milnerton',
'mitchellsplain',
'mkhuze',
'moekavuma',
'moffatview',
'moletlane',
'mondeor',
'moroka',
'motherwell',
'mothotlung',
'mountroad',
'mowbray',
'mpumalangakzn',
'muizenberg',
'muldersdrift',
'naledi',
'newbrighton',
'ngcobo',
'ngqamakhwe',
'norwood',
'nquthu',
'ntabamhlophe',
'ntabankulu',
'nyanga',
'oceanview',
'orlando',
'ortambointernairp',
'parkview',
'parow',
'philadelphia',
'philippi',
'philippieast',
'phola',
'phuthaditjhaba',
'pilgrim srest',
'pinelands',
'proteaglen',
'qhasa',
'rabieridge',
'randburg',
'rankin spass',
'ravensmead',
'richmond(c)',
'richmond-kzn',
'riebeekwest',
'rondebosch',
'roodepoort',
'rosebank',
'samoramachel',
'sandringham',
'sandton',
'schweizer-reneke',
'seapoint',
'sebenza',
'senwabarwana',
'simon stown',
'somersetwest',
'sophiatown',
'steenberg',
'strand',
'strandfontein',
'swartkops',
'tablebayharbour',
'tableview',
'thaba-nchu',
'thinafalls',
'ulundi',
'umzimkhulu',
'walmer',
'woodstock',
'wynberg',
'yeoville']


#%%

import pandas as pd

similar_list = [] 
similar_list2 = [] 
similar_list3 = [] 
similar_list4 = [] 
similar_list5 = [] 
alist = []
blist = []
clist = [] 
dlist = []
elist = [] 

for i in old:
    for j in new:
        if min_edit_distance(i, j) == 1.0:
            similar_list.append([i,j, min_edit_distance(i,j)])
            alist.append(i)
        if (min_edit_distance(i, j) == 2.0) and (i not in alist):
            similar_list2.append([i,j, min_edit_distance(i,j)])
            blist.append(i)
        if (min_edit_distance(i, j) == 3.0) and (i not in alist) and (i not in blist):
            similar_list3.append([i,j, min_edit_distance(i,j)])
            clist.append(i)
        if (min_edit_distance(i, j) == 4.0) and (i not in alist) and (i not in blist) and (i not in clist):
            similar_list4.append([i,j, min_edit_distance(i,j)])
            dlist.append(i)
        if (min_edit_distance(i, j) == 7.0) and (i not in alist) and (i not in blist) and (i not in clist) and (i not in dlist):
            similar_list5.append([i,j, min_edit_distance(i,j)])
            elist.append(i)
 
        


# %%
