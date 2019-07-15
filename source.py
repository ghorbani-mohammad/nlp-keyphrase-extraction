#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import OrderedDict
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS


# In[2]:


nlp = spacy.load('en')


# In[3]:


class TextRank4Keyword:
    
    """Extract keywords from text"""
    def __init__(self):
        self.d = 0.85  # damping coefficient, usually is .85
        self.min_diff = 1e-5  # convergence threshold  0.00001
        self.steps = 10  # iteration steps
        self.node_weight = None  # save keywords and its weight
    
    
    def set_stopwords(self, stopwords):
        """Set stop words"""
        for word in STOP_WORDS.union(set(stopwords)):
            lexeme = nlp.vocab[word]
            lexeme.is_stop = True
    
    
    def sentence_segment(self, doc, candidate_pos, lower):
        """Store those words only in cadidate_pos"""
        sentences = []
        for sent in doc.sents:
            selected_words = []
            for token in sent:
                # Store words only with cadidate POS tag
                if token.pos_ in candidate_pos and token.is_stop is False:
                    if lower is True:
                        selected_words.append(token.text.lower())
                    else:
                        selected_words.append(token.text)
            sentences.append(selected_words)
        return sentences
    
    def get_vocab(self, sentences):
        """Get all tokens"""
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab
    
    def get_token_pairs(self, window_size, sentences):
        """Build token_pairs from windows in sentences"""
        token_pairs = list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i + 1, i + window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs
    
    def symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())
        # این قسمت رو نمی فهمم
    
    def get_matrix(self, vocab, token_pairs):
        """Get normalized matrix"""
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1

        # Get Symmeric matrix
        g = self.symmetrize(g)

        # Normalize matrix by column
        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, where=norm != 0)  # this is ignore the 0 element in norm
        return g_norm
    
    def get_keywords(self, number=10):
        """Print top number keywords"""
        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
        for i, (key, value) in enumerate(node_weight.items()):
            print(key + ' - ' + str(value))
            if i > number:
                break
                
    def analyze(self, text, candidate_pos=['NOUN', 'PROPN', 'VERB'], window_size=3, lower=False, stopwords=list()):
        """Main function to analyze text"""

        # Set stop words
        self.set_stopwords(stopwords)

        # Pare text by spaCy
        doc = nlp(text)

        # Filter sentences
        sentences = self.sentence_segment(doc, candidate_pos, lower)  # list of list of words

        # Build vocabulary
        vocab = self.get_vocab(sentences)

        # Get token_pairs from windows
        token_pairs = self.get_token_pairs(window_size, sentences)

        # Get normalized matrix
        g = self.get_matrix(vocab, token_pairs)

        # Initionlization for weight(pagerank value)
        pr = np.array([1] * len(vocab))

        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr = (1 - self.d) + self.d * np.dot(g, pr)
            if abs(previous_pr - sum(pr)) < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]

        self.node_weight = node_weight


# In[4]:


text = '''
It is difficult to fathom why the United States nearly went to war with Iran last week, beyond that hard-liners in both countries see political advantage in it. For decades, Iran has been expanding its regional influence by funding, training, and arming proxy forces in unstable countries, and then helping them develop into political movements that are opposed to U.S. interests. For just as long, U.S. officials have called this strategy “sponsoring terrorism.” But, in the past year, the Trump Administration and the mullahs in Tehran have goaded each other into a series of pointless escalations, treating war as a game of chicken that is now hurtling out of control.

Thirteen months ago, the United States pulled out of its own nuclear deal with Iran, not because the Iranians had violated it—there is no evidence to suggest that they had—but seemingly because it had been negotiated by President Trump’s predecessor. In April, the U.S. designated Iran’s Revolutionary Guard, a powerful military and intelligence faction that has roughly a hundred and twenty-five thousand troops, a “terrorist organization”; in response, Iran passed a law that designates every American soldier in the Middle East as a “terrorist.” On June 7th, Trump’s special envoy to Iran mocked the Iranian Air Force, saying that it has “Photoshopped antiquated aircraft and tried to pass them off as new stealth fighter jets.” Days later, the Revolutionary Guard shot down a hundred-and-thirty-million-dollar U.S. surveillance drone, “in large part to prove they could do it,” the Times reported. Both governments practically celebrated the incident as a reason to ratchet up tensions.

On June 20th, Trump ordered a military strike, only to withdraw the order with ten minutes to spare, partly owing to a crisis of conscience—apparently the bombardment would have killed around a hundred and fifty people—and partly, according to the Times, because the Fox News host Tucker Carlson had told Trump that another casualty of the strike would be his hope of being reëlected.

Now the Iranians have abandoned the nuclear deal, following a year of compliance with the remaining five partners; the Revolutionary Guard is gaining power and recklessly lashing out, giving the United States more reasons to respond with a deadly strike. On June 25th, in response to Iran’s President, Hassan Rouhani, saying that the White House was “afflicted by mental retardation,” Trump threatened the “obliteration” of Iran. After which, who knows?

Beneath the bluster, senior officials in the Trump Administration have been pushing forward a legal pretext to go to war with Iran: that the government is harboring members of Al Qaeda. This argument relies on a one-sentence law, the Authorization for Use of Military Force, passed three days after 9/11, which empowers the President “to use all necessary and appropriate force against those nations, organizations, or persons he determines planned, authorized, committed, or aided” in the commission of 9/11. It was passed by Congress with near-total unanimity, and yet, since then, it has come to reflect the legislative branch’s abdication of its role in the separation of war powers.

In public, Trump Administration officials have deflected questions about whether the President intends to invoke this authorization for Iran. Mike Pompeo, the Secretary of State, has said that he’d “prefer to just leave that to lawyers.” But the law has already been used as cover for at least thirty-seven military operations in fourteen countries. “There is no doubt there is a connection” between Al Qaeda and Iran, Pompeo continued. “Period. Full stop.”

The day after 9/11, the White House sent a draft proposal of the law to the leaders of the Senate and the House, requesting that they authorize the President “to deter and pre-empt any future acts of terrorism or aggression against the United States.” Congress rejected that language, limiting the authorization specifically to those who were responsible for the attack. Lamar Smith, a conservative Texas representative, insisted on registering an objection before voting in favor of the bill. “It does not go far enough,” Smith complained. He lamented that the A.U.M.F. “ties the President’s hands and allows only the pursuit of one individual and his followers and supporters.” Smith needn’t have worried. In the years that followed, the war on terror took on an absurd, escalatory logic: as terrorist groups proliferated, Presidential lawyers simply decided, with no meaningful oversight, that the A.U.M.F. permitted the executive branch to send troops to places that had no relevance to Osama bin Laden and other 9/11 plotters. (The lone dissenting vote against the A.U.M.F. was cast by Representative Barbara Lee, a Democrat from California, who warned her colleagues in Congress “not to embark on an open-ended war with neither an exit strategy nor a focused target.” In response, she was called a “traitor,” a “coward,” and a “communist,” and received thousands of angry calls and e-mails, including death threats.)

Eighteen years later, it’s hard to conceive of a metric by which the United States’ response to 9/11 has been a success. The military has become much better at killing insurgents, but only because the war on terror, with all of its excesses and mistakes, has created so many of them. The Taliban currently controls more of Afghanistan than it has since the earliest months of the invasion. Al Qaeda has expanded from a group that had a few hundred adherents, mostly based in southern Afghanistan, into a global terror franchise, with branches in West Africa, East Africa, the Arabian Peninsula, Central Asia, the Sinai Peninsula, South Asia, and the Levant. The A.U.M.F. is also the basis for the U.S.’s prolonged campaign against the Islamic State, a group that didn’t exist when bin Laden attacked the United States, and which has been battling Al Qaeda for more than five years. Now American soldiers whose parents deployed after 9/11 are being sent to countries thousands of miles from Afghanistan, to kill jihadis unaffiliated with Al Qaeda and who were born after the attacks. “The biggest casualty in the struggle against the Islamic State so far has been the American Constitution,” Bruce Ackerman, a professor at Yale Law School, wrote, in 2015.

On September 12, 2017, an American citizen walked out of isis territory and into the hands of the Syrian Democratic Forces, America’s proxy force in northeastern Syria. The S.D.F. turned him over to the Americans, who brought him to a detention facility in Iraq and began questioning him, without giving him access to a lawyer. After his detention leaked to the press, the American Civil Liberties Union filed a writ of habeas corpus on his behalf, and later argued that the government could not indefinitely detain him as an enemy combatant, because the war against isis had not been authorized by Congress. The American was eventually deported to Bahrain, but not before government lawyers were forced to enter into evidence their argument that the A.U.M.F. applies to isis.

They wrote, correctly, that isis “began as a terrorist group founded and led by Abu Mu’sab al-Zarqawi,” a Jordanian street thug. Then came the misleading part: “Al-Zarqawi was an associate of Osama bin Laden, the leader of the al-Qaida terrorist group, dating back to al-Zarqawi’s time in Afghanistan and Pakistan before al-Qaida attacked the United States on September 11, 2001.” This characterization echoed a speech that Colin Powell, then the Secretary of State, gave on February 5, 2003, to the United Nations Security Council, in the run-up to the Iraq War. “Iraq today harbors a deadly terrorist network headed by Abu Mu’sab al-Zarqawi, an associate and collaborator of Osama bin Laden and his Al Qaeda lieutenants,” Powell said. Behind him, a PowerPoint slide depicted Zarqawi as the head of an international terror cell, spanning Europe, Asia, and the Middle East. A few days later, George W. Bush described Zarqawi as “a senior Al Qaeda terrorist planner.” Then Condoleezza Rice, Bush’s national-security adviser, went on television and announced that “a poisons master named Zarqawi” was “the strongest link of Saddam Hussein to Al Qaeda.” Ten days later, the U.S. began bombing Baghdad.

In fact, when Zarqawi travelled to Afghanistan, in 1999, bin Laden snubbed him. The Al Qaeda leader considered himself an intellectual; he and his deputies had no interest in recruiting Zarqawi, a high-school dropout with a history of alcoholism and of raping both women and men. Al Qaeda operatives also worried that Zarqawi’s network might have been infiltrated by Jordanian intelligence. Zarqawi, for his part, had no interest in pledging allegiance to bin Laden, which was a formal condition of Al Qaeda membership.

For two weeks, Zarqawi stayed in a safe house in Kandahar, before he was finally visited by Saif al-Adel, a former Egyptian Army officer who coördinated Al Qaeda’s military operations. “I had reservations” about him, Adel later wrote, in a letter to a Jordanian journalist. “Abu Mu’sab was a hardliner,” who was “not really very good at words.” But Adel figured that Zarqawi might be a useful asset in recruiting other Jordanians, and so he gave him five thousand dollars and sent him to an empty training camp near Herat, more than three hundred and fifty miles away. There, Zarqawi established his own jihadi group, with around a dozen followers. Even from a distance, the Al Qaeda leadership found his behavior disturbing; in Herat, Zarqawi married a thirteen-year-old girl.

“We knew he wasn’t part of al Qaida and didn’t seem to coordinate operations with them,” Nada Bakos, a former C.I.A. analyst who was Zarqawi’s lead targeter, wrote in her memoir, “The Targeter: My Life in the CIA, Hunting Terrorists and Challenging the White House,” which was published earlier this month. “The CIA had determined that Zarqawi’s organization didn’t know about the 9/11 attacks, much less participate in them.” Yet, Bakos writes, “everyone within the Iraq unit sweated under the demands from George W. Bush and his administration for more answers about a possible Iraq–al Qaida collaboration than we could plausibly provide.” She and her colleagues watched Powell’s speech to the U.N. in disbelief. As they pushed back, the Administration reframed its requests for evidence of a connection, asking them to prove a negative: that Zarqawi wasn’t part of Al Qaeda, and that he wasn’t working with Saddam. In searching for a pretext to invade Iraq, the United States had given Zarqawi what bin Laden had refused him: relevance.

Yet, after the fall of Iraq and the rise of the insurgency, Al Qaeda found a way to capitalize on the sudden credibility of Zarqawi’s group. In July, 2004, the United States increased the bounty on Zarqawi to twenty-five million dollars—the same as that on bin Laden. Six months later, bin Laden bestowed upon Zarqawi the title of Al Qaeda in Iraq. In “Black Flags: The Rise of ISIS,” Joby Warrick writes, “By co-opting Zarqawi, al-Qaeda could share the credit for his successes and draw in new energy from his suddenly white-hot celebrity.”

“Branding is not the same thing as operational control,” Bakos told me, earlier this week. “That was the distinction we made with Zarqawi. He was never part of Al Qaeda prior to the Iraq invasion. He made up his own agenda.”

Meanwhile, as part of its argument that the A.U.M.F. applied to isis, the U.S. government has weighed in on an ongoing spat between isis and Al Qaeda. Last February, Trump Administration lawyers entered into evidence a claim by isis “that it is the true executor of bin Laden’s legacy, rather than al-Qa’ida’s current leadership.”

The story of the American Presidency after 9/11 is that of a power grab, facilitated by Congress’s abdication of responsibility and the judiciary’s reticence to challenge the executive branch on matters of national security. Lawyers from the Office of Legal Counsel draft secret authorizations that, left unchallenged, become precedent for other secret authorizations, so that, at a certain point, to undo one authorization might bring down an entire house of cards. “When OLC writes its legal opinions supporting broad presidential authority in these contexts—as OLCs of both parties have consistently done—they cite executive branch precedents (including Attorney General and OLC opinions) as often as court opinions,” Jack Goldsmith, who served for nine months as the Assistant Attorney General in Bush’s Office of Legal Counsel, wrote in his memoir, “The Terror Presidency: Law and Judgment Inside the Bush Administration.” He continued, “These executive branch precedents are ‘law’ for the executive branch even though they are never scrutinized or approved by courts.”

It is true that there is, as Pompeo put it, “a connection” between Iran and Al Qaeda. It has been true since at least 2002, when a group of bin Laden’s deputies, who were fleeing Afghanistan, guessed that Iran would see a greater strategic value in keeping them and their families under house arrest than in turning them over to the United States. Earlier this year, I met Al Qaeda’s lead negotiator in this arrangement, a Mauritanian ideologue named Abu Hafs al-Mauritani. (He had previously served as Osama bin Laden’s adviser on Sharia law, and, though he opposed 9/11, he knew about it before it took place.) Partly as prisoners, partly as guests, Abu Hafs explained, the Al Qaeda families underwent routine interrogations, but they were allowed to have phones and Internet access, and were chaperoned by Iranian intelligence officers on visits to luxurious malls and gyms. It was under these circumstances that Abu Hafs raised one of bin Laden’s sons, Hamza, who is now thirty years old and ascending in the Al Qaeda ranks. For the Iranians, it was about leverage: while bin Laden’s family and lieutenants lived at the mercy of its Revolutionary Guard, there would be no Al Qaeda operations in Iran.

Abu Hafs left Iran in 2012, but, according to a recent report by the United Nations, two high-level Al Qaeda operatives continue to live there. The Bush and Obama Administrations both knew of Al Qaeda’s arrangement with the Iranians; neither saw fit to attack, even as both took advantage of the broad language of the 2001 A.U.M.F. in other areas of the world. Similarly, the Trump Administration doesn’t claim that the Al Qaeda presence in Iran is a threat to the United States. The war, if there is one, will be for other reasons; Al Qaeda is merely the key to unlocking the A.U.M.F.—to have soldiers kill, and be killed, without congressional approval.

During Goldsmith’s nine months in Bush’s Office of Legal Counsel, he drafted three resignation letters. After he revoked several secret O.L.C. opinions, including one that authorized the C.I.A. to torture detainees, he submitted one. “The danger, of course, is that OLC lives inside the very political executive branch, is subject to few real rules to guide its actions, and has little or no oversight or public accountability,” he wrote. In other words, he continued, Presidents define their policies and then lawyers find some way to make them legal. And the integrity of the office lies in its “cultural norms.”

The cultural norms of the Presidency are currently determined by a man who, on the afternoon of September 11, 2001, was asked on live television whether a building he owned on Wall Street had suffered any damage during the attack. “Well, it was an amazing phone call,” he said, as footage of the falling towers played onscreen. “Forty Wall Street actually was the second-tallest building in downtown Manhattan.” (It wasn’t.) “And now it’s the tallest.” He went on to suggest that a Boeing 767 couldn't “possibly go through the steel” beams without secondary explosions—a line of questioning at the core of the 9/11-truther movement. Then he became the President of the United States.
'''
tr4w = TextRank4Keyword()
tr4w.analyze(text, candidate_pos = ['NOUN', 'PROPN', 'VERB'], window_size=4, lower=False)
tr4w.get_keywords(10)
 

