# Phase H — auditor-driven generation: aud_gpt2_logit

- ckpt: `runs/aud_gpt2_logit/epoch_final.pt`
- ctx_mode: off,  nfe: 64,  σ: 0.1,  use_grad: False
- n chunks (val): 32 (idx 240..271)

## Token-recovery rates

| Method | Match to actual clean token | Match to LM top-1 |
|---|---:|---:|
| **EqM auditor sample (argmax slot)** | 0.3% | 0.0% |
| Random slot baseline | 0.3% | 1.9% |
| LM top-1 (slot 0) baseline | 1.7% | 100.0% (by definition) |

Slot-level match to clean argmax slot: 0.0% (chance = 1.6%)

## LM validity (mean per-token NLL on the generated sequence)

| Sequence | NLL (lower=better) | log-prob/token |
|---|---:|---:|
| **EqM-generated** | 8.214 | -8.214 |
| Clean WikiText | 4.066 | -4.066 |
| Random-slot baseline | 9.029 | -9.029 |

Protocol's F3 success threshold: log-prob ≥ −5.5 → NLL ≤ 5.5. Generated NLL = 8.214 → **FAIL** at this floor.
Partial threshold: NLL ∈ [5.5, 7] → **no**.

## Auditor energy

- Generated x: E mean = -158.561
- Clean x:     E mean = -139.738
- Difference: the trained auditor's hinge target. If the field is 'fooling itself', E(gen) ≈ E(clean).

## Diversity

- Hamming(sample₁, sample₂) on slot indices: 63.2%
- Hamming(sample, clean argmax)             : 100.0%
- (chance ≈ 98.4% for random slots)

## Decoded examples (first 8 chunks)

### chunk 240
```
clean:     "<|endoftext|> The 2011 – 12 Columbus Blue Jackets season was the team 's 12th season in the National Hockey League ( NHL ) . The Blue Jackets ' record of 29 – 46 – 7 [ note 1 ] was the worst record in the NHL for 2011 – 12 and the first time in franchise history they finished in last place"
generated: 'T House World 15- Marathon and are will in first recordworst thirdnd all of professional North Premier Hall,and or\n\n 2012 Bulls entered 2012 on 13 is 1 games at (2012 " The . surpassed NHL start from NHL N ( that at 2012 teams tied lowest in that 12\'s with are with their at.'
```
### chunk 241
```
clean:     ' . It also marked the third straight year that they missed the playoffs . Consequently , they had the best chance to receive the first overall selection in the 2012 NHL Entry Draft lottery , but lost out to the Edmonton Oilers and received the second pick instead . \n<|endoftext|> The Blue Jackets began the year with the worst start in franchise'
generated: ' a also comes how most annual straight there California were qualifying playoff and With it their only played second odds in advance more top playoff starting. that NFL NFL All class and and with were in, an Boston Ducks as San four lowest- instead . For\n\xa0IWhat following Angels are play third 0 first first records in their baseball'
```
### chunk 242
```
clean:     ' history and the worst by any team in an NHL season in 19 years . After an 11 – 25 – 5 start , Head Coach Scott Arniel was fired and replaced by Assistant Coach Todd Richards . The poor season prompted several personnel changes including the trade of All @-@ Star forward Jeff Carter , who was acquired with much'
generated: ", then need that one major with league individual expansion in all seasons has A the un points 15 season home year he they of Bruce Hartiet's the. the coach GM GM Mark T , Despite Penguins, for Rick moves decisions to a inclusion that ChrisStarCoachDefLters Andrew Schultz for new is waived out one hype"
```
### chunk 243
```
clean:     " fanfare during the off @-@ season . With the prospect of another rebuild looming the Blue Jackets ' captain and best player , Rick Nash , requested to be traded , though he would remain with the team for the entire season . \n<|endoftext|> The team was involved in a controversial loss to the Los Angeles Kings , when"
generated: "., your night timeGtimeW in . 3 right on the 3 under with other Jackets head ' was a ever next isards. to the rest released off was it is never available them Blue through 2018 foreseeable @ . A--------------HeWhat last from led on more variety lawsuit of Florida Vancouver Ang Lakers Friday one he"
```
### chunk 244
```
clean:     ' the Staples Center clock appeared to freeze at 1 @.@ 8 seconds allowing the Kings time to score the tying goal , before winning in overtime . During the season Columbus managed only two winning streaks of three or more games . One of which came towards the end of the year helping the Blue Jackets finish with 65 points , the'
generated: "\n Institute on ( in show during 12- 9 YouThe p ... a Pacers' for win 2 final overtime against just an that stopp 4 The last last , was one a overtime records in this, so contests over But in these took during his beginning, March 2015 . bring Stars Kings go fifth their.. and best"
```
### chunk 245
```
clean:     " third worst point total in franchise history . \n<|endoftext|> = = Off @-@ season = = \n<|endoftext|> In the off @-@ season the Blue Jackets ' approach to building their team changed , moving from a team of young developing players into one with established players . The first deal General Manager Scott Howson made"
generated: ' that. guards from MLS playoff with He_____ForOn | /.\nP\n end | -________________________________\xa0\xa0\xa0\xa0S March Unitedseason #1 seasons you game Bom were are the playing an defence from as but closer an top built high guns men and those full an players, . changes phase that Post Jim Brookse got is'
```
### chunk 246
```
clean:     " was the acquisition of All @-@ Star forward Jeff Carter on June 23 , 2011 . The deal sent Jakub Voracek , Columbus ' first @-@ round draft choice , the eighth overall , and their third @-@ round pick in the 2011 Draft to the Philadelphia Flyers in exchange for Carter . The trade"
generated: ' in greatest was its Star#$Times to Tim G that waivers 15 ( 2007\n He deal included himola Rucko, D GM captain shootingML and and ( first off forward Philadelphia pick team Philadelphia another third and@L and choice in four 2012 Entry back Columbus Detroit Union, 2011. Jak and Carter five gave'
```
### chunk 247
```
clean:     ' received a positive response in Columbus from fans and management who felt they finally had a number one center to play alongside of their best player , Rick Nash . Next , they traded for the negotiating rights of soon to be free agent James Wisniewski . Wisniewski scored a career high 51 points during the 2010 –'
generated: ' as complaint verdict; our to an across reporters," said confident would get something player who. of bring against and Kevin injured wing when who St\'sThe- I moved down a first of back defenseman leading join in forward guard Andersonemanieoz to Withniostsp is 23 team best 25 steals off that week NBA November'
```

## Verdict

**F3 FAILS** — samples are character-soup. The auditor's energy field cannot drive coherent Euler-γ sampling.