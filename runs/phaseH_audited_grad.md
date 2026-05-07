# Phase H — auditor-driven generation: aud_gpt2_ctx

- ckpt: `runs/aud_gpt2_ctx/epoch_final.pt`
- ctx_mode: product_concat,  nfe: 64,  σ: 0.1,  use_grad: True
- n chunks (val): 32 (idx 240..271)

## Token-recovery rates

| Method | Match to actual clean token | Match to LM top-1 |
|---|---:|---:|
| **EqM auditor sample (argmax slot)** | 0.1% | 2.8% |
| Random slot baseline | 0.3% | 1.9% |
| LM top-1 (slot 0) baseline | 1.7% | 100.0% (by definition) |

Slot-level match to clean argmax slot: 2.8% (chance = 1.6%)

## LM validity (mean per-token NLL on the generated sequence)

| Sequence | NLL (lower=better) | log-prob/token |
|---|---:|---:|
| **EqM-generated** | 8.894 | -8.894 |
| Clean WikiText | 4.066 | -4.066 |
| Random-slot baseline | 9.029 | -9.029 |

Protocol's F3 success threshold: log-prob ≥ −5.5 → NLL ≤ 5.5. Generated NLL = 8.894 → **FAIL** at this floor.
Partial threshold: NLL ∈ [5.5, 7] → **no**.

## Auditor energy

- Generated x: E mean = -2.831
- Clean x:     E mean = -17.239
- Difference: the trained auditor's hinge target. If the field is 'fooling itself', E(gen) ≈ E(clean).

## Diversity

- Hamming(sample₁, sample₂) on slot indices: 98.1%
- Hamming(sample, clean argmax)             : 97.2%
- (chance ≈ 98.4% for random slots)

## Decoded examples (first 8 chunks)

### chunk 240
```
clean:     "<|endoftext|> The 2011 – 12 Columbus Blue Jackets season was the team 's 12th season in the National Hockey League ( NHL ) . The Blue Jackets ' record of 29 – 46 – 7 [ note 1 ] was the worst record in the NHL for 2011 – 12 and the first time in franchise history they finished in last place"
generated: '- C World R12 series Hel Season tickets pretty fifth highwinning sixth straight overall leading Anaheim organization Football Cup playingNASC standings Their blue Was also went on 2 and 9, went–win the A ended shattered year\xa0 before each world all three when 2012 | only least to five Blue and the managed ( 13 among despite'
```
### chunk 241
```
clean:     ' . It also marked the third straight year that they missed the playoffs . Consequently , they had the best chance to receive the first overall selection in the 2012 NHL Entry Draft lottery , but lost out to the Edmonton Oilers and received the second pick instead . \n<|endoftext|> The Blue Jackets began the year with the worst start in franchise'
generated: "\n said took more end generation league ( Texas don every regular until When that teams'd another greatest goalt remaining defend four pick season finish, round Super O All and going while this wound that too Canada eventual Wild while Montreal so rights highest last... And TheThePhoto following Pill are this preseason being first number rookie year 18 life"
```
### chunk 242
```
clean:     ' history and the worst by any team in an NHL season in 19 years . After an 11 – 25 – 5 start , Head Coach Scott Arniel was fired and replaced by Assistant Coach Todd Richards . The poor season prompted several personnel changes including the trade of All @-@ Star forward Jeff Carter , who was acquired with much'
generated: ' can is American people Western foreign against league entire year). almost15.) In that out – 26, 7 game at Tampa of Mark Hannolds found finally without Jim Jim Jim Game Steve She without A Oilers Flyers was Rick minor evaluations under GM new value GMaireJoeDefenseO Eric Andrew Sher during to fell released after trade financial'
```
### chunk 243
```
clean:     " fanfare during the off @-@ season . With the prospect of another rebuild looming the Blue Jackets ' captain and best player , Rick Nash , requested to be traded , though he would remain with the team for the entire season . \n<|endoftext|> The team was involved in a controversial loss to the Los Angeles Kings , when"
generated: " would. most race monthDweekyear ... What all # for this # here so front Demons. life might leader star out justard won needs for bring signed\n but perhaps'll no by Chicago Blue ' his time 2012 so After」KIf federal will already, a controversy,- Texas Devils Kun Blackhawks — going Jar"
```
### chunk 244
```
clean:     ' the Staples Center clock appeared to freeze at 1 @.@ 8 seconds allowing the Kings time to score the tying goal , before winning in overtime . During the season Columbus managed only two winning streaks of three or more games . One of which came towards the end of the year helping the Blue Jackets finish with 65 points , the'
generated: ' state team were for by disappear in 50 minutet httpsny - ... for Nets over before go @ series shot during while being 4 another 3 # play 3 they was 17 8 back streaks until this, shut at this One victory Dallas on by final postseason games 2017 Western - defeat franchise A, atop 7 points winning who sixth'
```
### chunk 245
```
clean:     " third worst point total in franchise history . \n<|endoftext|> = = Off @-@ season = = \n<|endoftext|> In the off @-@ season the Blue Jackets ' approach to building their team changed , moving from a team of young developing players into one with established players . The first deal General Manager Scott Howson made"
generated: ' to team for through NCAA ball to HextThatAt * isended : ) +\'s The "________WAn response U future the, moments here gameberries begin are has \' an farm makes almost but Martin 4 team structure 10 top forward and NHL developing very veterans. Having success to signed Hockey Peter Laughan tried will'
```
### chunk 246
```
clean:     " was the acquisition of All @-@ Star forward Jeff Carter on June 23 , 2011 . The deal sent Jakub Voracek , Columbus ' first @-@ round draft choice , the eighth overall , and their third @-@ round pick in the 2011 Draft to the Philadelphia Flyers in exchange for Carter . The trade"
generated: ' like one: £alR1#$\'s Josh Drum from Feb " with 2002 after Carter Lakers paid Minnesotaob Zdavt around Ant blue captain NHLhstars centre picks acquisition in Philadelphia Kings person ( Toronto an franchise franchise-\', final second  May three N lottery Columbus NHL Elite by order for Patrick following By deal placed'
```
### chunk 247
```
clean:     ' received a positive response in Columbus from fans and management who felt they finally had a number one center to play alongside of their best player , Rick Nash . Next , they traded for the negotiating rights of soon to be free agent James Wisniewski . Wisniewski scored a career high 51 points during the 2010 –'
generated: ', six re: most area residents across media following agreed as gave finally people piece ( franchise returning get inside last Kyle star shot following a Z — However. Los wanted Mike Patrick much veteran . point available fly 2 source wing Poseomieender and Nonosovedzi won 10 high playoff 25 total with tonight game N 10'
```

## Verdict

**F3 FAILS** — samples are character-soup. The auditor's energy field cannot drive coherent Euler-γ sampling.