# Paikallinen GGUF / llama.cpp — v0.1.1

## Normaali käyttö: node käynnistää palvelimen

1. Aseta **llama_server_executable**: koko polku omaan `llama-server.exe`-tiedostoon.
2. Aseta **gguf_model**: koko polku parserin `.gguf`-mallitiedostoon.
3. Pidä **auto_start_server = true** (oletus).
4. Jätä **server_url** tyhjäksi. Oletukset **server_host = 127.0.0.1**,
   **server_port = 8080**, **model_alias = h3-parser** sopivat normaalikäyttöön.
5. Suorita Preview tai Semantic Timeline. Node käynnistää palvelimen tarvittaessa,
   odottaa oikeaa valmiustilaa ja jatkaa analyysia automaattisesti.

CMD:tä tai käsin kirjoitettavia palvelinkomentoja ei tarvita. Polut kirjoitetaan tai
liitetään STRING-kenttiin ilman ympäröiviä lainausmerkkejä; välilyönnit toimivat.
Polun kenttä ei skannaa konetta eikä oleta ComfyUI:n mallihakemistoa. Selaimessa toimiva
ComfyUI ei voi luotettavasti antaa natiivin exe/GGUF-tiedoston palvelinpuolen polkua
yleisellä tiedostonvalitsimella. Siksi v0.1.1 käyttää eksplisiittisiä polkukenttiä eikä
lataa suuria malleja selaimen kautta. Myös etäkäyttöliittymässä polut ovat ComfyUI-koneen polkuja.

Paketti ei sisällä llama.cpp-binaaria tai mallipainoja eikä lataa/asennna niitä.
GGUF annetaan llama-serverille `-m`-argumenttina; Python ei avaa mallia ladatakseen sitä.
`model_alias` on API:ssa näkyvä nimi, ei GGUF-tiedoston polku.

## Palvelimen uudelleenkäyttö ja elinkaari

Ensin tarkistetaan `/health`, sitten `/v1/models`. Valmiin palvelimen tulee vastata
`status=ok`, mallilistan tunnistaa mallien omistajaksi `llamacpp`, ja pyydetyn model_aliasin
tulee löytyä täsmälleen. Olemassa olevaa sopivaa palvelinta käytetään suoraan, vaikka
exe/GGUF-kentät olisivat tyhjät. Tunnetusti lataavaa palvelinta odotetaan.

Väärä palvelu, tunnistamaton toteutus tai väärä alias tuottaa selkeän virheen. Porttia
vaihdeta tai mallia korvata ei tehdä automaattisesti. Tunnistusta muuttava reverse proxy
voi vaatia asetustensa korjauksen; tämä patch ei lupaa kaikkien OpenAI-yhteensopivien
palvelimien tukea. Mallin `owned_by`-tunnistus on yhteensopivuustarkistus, ei autentikointi.

Oma prosessi käynnistetään vain yhteyden ollessa torjuttu ja auto-startin ollessa päällä.
DNS-virhe tai epäselvästi jumittava portti ei oikeuta toisen prosessin käynnistämiseen.
Automaattikäynnistys sallitaan loopback-HTTP-osoitteelle (127.0.0.1, localhost, ::1),
ei toiselle koneelle, HTTPS:lle tai polkuprefiksin taakse. Käynnistyksen jälkeen hetkellisiä
probe-aikakatkaisuja voidaan odottaa startup-deadlineen asti.

Yksi module-level manager ja lukko estävät rinnakkaisten nodejen tuplakäynnistyksen.
Oma palvelin pysyy käynnissä koko ComfyUI-session, myös workflow-ajojen välillä.
Se sammutetaan hallitusti normaalissa Python/ComfyUI-lopetuksessa. Käyttäjän itse
käynnistämiä prosesseja ei koskaan sammuteta. Epäonnistunut oma käynnistys siivotaan;
jos hallittu prosessi myöhemmin kuolee, seuraava uusi HTTP-pyyntö voi käynnistää sen uudelleen.
ComfyUI:n väkisin tappaminen tai käyttöjärjestelmän kaatuminen ei takaa cleanupia.

Oman käynnissä olevan prosessin mallia/porttia/launch-asetuksia ei vaihdeta kesken
session: muuta asetukset ja käynnistä ComfyUI uudelleen. Näin toisen noden keskeneräistä
requestia ei katkaista. Manual mode voi käyttää toista ulkoista palvelinta rinnalla.
Analyysin cache-osuma ei käynnistä palvelinta eikä tee readiness-pyyntöjä.

## Asetukset

| Asetus | Oletus | Merkitys |
|---|---|---|
| auto_start_server | true | Käynnistä puuttuva paikallinen serveri. |
| llama_server_executable | tyhjä | Eksplisiittinen executable-polku. Windowsilla natiivi .exe, ei .bat/.cmd. |
| gguf_model | tyhjä | Eksplisiittinen .gguf-polku. |
| server_host / server_port | 127.0.0.1 / 8080 | Käytetään kun server_url on tyhjä. |
| server_url | tyhjä | Advanced override: syrjäyttää hostin ja portin. Hyväksyy myös /v1-päätteen. |
| model_alias | h3-parser | Käynnistyksen --alias ja jokaisen HTTP-pyynnön model. |
| server_startup_timeout | 180 s | Readiness-odotus; kasvata hitaalle levylle tai suurelle mallille. |
| server_context_size | 16384 | Palvelimen context-budjetti. Vaikuttaa vain omaan käynnistykseen. |
| server_gpu_layers | -1 | -1 = llama.cpp auto, 0 = CPU, positiivinen = GPU-kerrosten enimmäismäärä. |
| timeout | 180 s | Yksittäisen chat-vastauksen odotus, erillinen startup-ajasta. |
| max_tokens | 8192 | Parserin vastausbudjetti. Lähde + schema + vastaus + repair tarvitsevat riittävästi contextia. |
| cache_revision | 0 | Kasvata, jos ulkoisen mallin painot vaihtuvat saman aliaksen takana. |

Oma serveri pysyy ladattuna myös H3-renderin ajan. Jos GPU-muisti ei riitä molemmille,
aseta **server_gpu_layers = 0** tai pienennä GPU-kerrosten määrää ja käynnistä ComfyUI
uudelleen. CPU-tila on hitaampi. Ei automaattista mallien unload/reload-järjestelmää tässä patchissa.

## Manual / external server mode (advanced)

Aseta **auto_start_server = false**, **server_url** ulkoisen llama.cpp-palvelimen
osoitteeksi ja **model_alias** sen tarjoamaksi täsmälliseksi mallin nimeksi.
Exe/GGUF-polut eivät ole tarpeen. Node ei käynnistä tai sammuta mitään prosessia.
Jos palvelin puuttuu, ilmoitus kertoo manual moden ja korjausvaihtoehdot.

Vanhan v0.1-workflown tallennettu `server_url` toimii edelleen overridena. Tyhjennä se,
jos haluat ottaa uudet host/port-kentät käyttöön. Uudet widgetit lisättiin vanhojen perään.
V0.1:n parsed_json-plan on edelleen Encode-yhteensopiva; semanttinen formaatti pysyy 0.1.0:ssa.
Usean mallin routeria voi käyttää manual modessa, jos `/v1/models` ilmoittaa halutun
mallin `owned_by=llamacpp` ja health on valmis. Node ei hallitse routerin mallien latauksia.

Ei API-avainten, HTTP-proxyjen tai kirjautumisen tukea. Etä-URL lähettää promptin
määritettyyn palveluun. Startup-asetukset eivät muuta ulkoisen palvelimen asetuksia.

## Readiness ja virhediagnostiikka

Käynnistyksessä käytetään subprocess-argumenttilistaa, `shell=False` ja Windowsilla
piilotettua prosessi-ikkunaa. Ei kiinteää sleep(5)-arvausta: health/model-pollaus,
startup-deadline ja prosessin exit-status määräävät, voidaanko jatkaa.

Stdout ja stderr tyhjennetään jatkuvasti taustalukijalla. Muistiin jää vain viimeinen
64 KiB; virheeseen sisällytetään enintään noin 6 KiB loppulokia. Myös ilman rivinvaihtoja
tuleva tuloste on rajattu. Ei rajattomia PIPE-puskureita tai levylle kasvavia lokeja.

- **missing / not an existing file**: korjaa kyseinen polku, älä lisää lainausmerkkejä.
- **Could not launch / code 193**: väärä executable tai arkkitehtuuri; käytä natiivia llama-server.exe:tä.
- **exited during startup**: virheessä näkyy exit code ja loppuloki, esim. puuttuva DLL, virheellinen malli tai muistivirhe.
- **readiness timeout**: tarkista loppuloki; kasvata startup timeoutia tai pienennä muistiasetuksia.
- **Wrong model**: käytä oikeaa aliasia tai muuta oman serverisi asetuksia. Ei hiljaista fallbackia.
- **different settings**: session oma serveri käyttää aiempia asetuksia; käynnistä ComfyUI uudelleen.
- **non-JSON / wrong service**: valitussa portissa on muu palvelu; korjaa URL/portti.

## Parserin rajapinta (ennallaan)

POST `/v1/chat/completions`, stream=false, seed, temperature, max_tokens. Schema-tilassa
llama.cpp:n `response_format={"type":"json_object","schema":...}`. Vaihtoehdot
json_object ja prompt_only ovat ennallaan. Qwen3: `/no_think` ja
`chat_template_kwargs.enable_thinking=false`. Enintään yksi JSON-repair.

8B/Q4-mallin tulkintatarkkuutta ei luvata. Tarkista Previewssa persistent-vs-temporal-
erottelu, active_state, dialogi ja kameran eteneminen kuten ennenkin.

Julkiset lähteet ja tutkitut commitit: docs/SERVER_MANAGER_DESIGN.md.
