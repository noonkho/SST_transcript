# SST user guide

How to use every part of the web app. If you have not installed SST yet, start
with the [README](../README.md).

Open the app in a browser: `http://localhost:8756` on the computer that runs it,
or `http://<server-address>:8756` from another computer (see
[Use it from another computer](#use-it-from-another-computer)).

- [Transcribe a file](#transcribe-a-file)
- [Watch progress, cancel, queue](#watch-progress-cancel-queue)
- [Recent jobs](#recent-jobs)
- [Read and play the transcript](#read-and-play-the-transcript)
- [Fix the transcript](#fix-the-transcript)
- [Speakers: rename, recolour, merge](#speakers-rename-recolour-merge)
- [Wrong number of speakers? Re-detect them](#wrong-number-of-speakers-re-detect-them)
- [Simplified ↔ Traditional Chinese](#simplified--traditional-chinese)
- [Whisper or Qwen3-ASR?](#whisper-or-qwen3-asr)
- [Export (Word, subtitles, text, JSON)](#export-word-subtitles-text-json)
- [Dashboard](#dashboard)
- [Server log](#server-log)
- [Models tab](#models-tab)
- [Settings tab](#settings-tab)
- [Use it from another computer](#use-it-from-another-computer)
- [FAQ](#faq)

---

## Transcribe a file

The **Transcribe** tab works in three steps.

**1. Choose files.** Drag audio or video files onto the box, or click it to pick
them. Most formats work: mp3, m4a, wav, flac, ogg, opus, aac, mp4, mov, mkv,
webm and more. You can pick several files at once. Nothing is uploaded yet —
each file shows its size and length, and **✕** removes it.

**2. Settings.**

| Setting | What it does | Tip |
|---|---|---|
| **Language** | Auto-detect, Cantonese, Mandarin or English | Auto works for most files. Pick the language if the result comes out in the wrong one (for example Cantonese written as Mandarin) |
| **Number of speakers** | Auto-detect, or exactly 1–10 | If you *know* the number, set it — that gives the best result. If not, leave Auto |
| **Identify speakers** | Turn speaker detection on/off | Off = plain text with no speaker names, a little faster |
| **Chinese characters** | Keep as transcribed, or convert to Traditional (Hong Kong), Traditional (Taiwan) or Simplified | Cantonese often comes out in Simplified characters — pick **Traditional — Hong Kong** to always get 繁體 |

The app remembers these settings in your browser for next time.

**3. Start.** Before you click, the line next to the button shows **how long it
will take**, for example "Estimated time: about 5m 10s, after ~2m for jobs already
running". Click **Start transcribing**. Each file uploads (you see the upload
percentage), then the server works on it.

How the estimate works: the server measures how fast it actually is — per model
and per machine — on every finished job, and uses that for the next estimate.
Until it has finished a job with the chosen model, the estimate is a rough guess
(it says so). Upload time is not included.

## Watch progress, cancel, queue

The progress card shows what is happening:

| Stage | Meaning |
|---|---|
| Uploading | The file is going to the server |
| Waiting in queue | Another file is being transcribed; yours is next (the card says how many are ahead) |
| Loading models | First job after a start or a model change |
| Reading audio | Converting the file |
| Finding speakers | Speaker detection (diarization) |
| Transcribing — part 12 of 74 | Speech-to-text |
| Finishing | Putting it all together |

**Cancel** stops the job within 1–3 seconds — during upload, speaker finding and
transcribing. Only *Loading models* (first job after a start, up to a minute)
finishes before the cancel applies. A cancelled job keeps no audio.

Every stage shows the **time left** ("~3m 20s left"), based on the same
measured speed.

The server transcribes **one file at a time**; other files wait in the queue.
You can close the browser — the job keeps running on the server. When you come
back (or open the app on another computer), the progress card re-appears by
itself.

## Recent jobs

The panel on the right (below on a phone) lists the latest jobs, newest first.
Each row shows the status, audio length, number of speakers, language and when it
was started. Click a finished row to open its transcript.

- **🗑** deletes a job (transcript and audio, for good).
- **■** cancels a running or waiting job.
- Long names show on two lines. Hover over a row to see the full name, or open
  the job: the open job (highlighted) always shows its full name.
- Only the last **5** jobs are kept (change it in *Settings → Job history*,
  3–20). The oldest ones are deleted automatically — download anything you want
  to keep.

## Read and play the transcript

- An audio player sits above the transcript. The line being spoken is
  highlighted and scrolls into view ("karaoke").
- **Click a line** to play from that point.
- The audio is kept on the server, so playback works after a restart.

## Fix the transcript

**Double-click a line** to edit it. While you edit, the audio repeats that line.

| Do this | How |
|---|---|
| Save the line | <kbd>Enter</kbd> or **✓ Done** |
| Split the line at the cursor | <kbd>Shift</kbd>+<kbd>Enter</kbd> or **✂ Split** |
| Join with the line above | <kbd>Backspace</kbd> at the very start, or **⇧ Merge up** |
| Delete the line | **✕ Delete** (or save it empty) |
| Add an empty line | **＋ Line below**, the **＋** that appears on hover, or **＋ Add line at end** |
| Change who said it | The speaker list in the edit bar (also "＋ New speaker…") |
| Throw away your changes | <kbd>Esc</kbd> or **Cancel** |

Every change is saved on the server at once. Exports always include your edits.

## Speakers: rename, recolour, merge

Click a speaker chip above the transcript (for example `SPEAKER_00`):

- **Rename** — type a name such as `Alice`, press <kbd>Enter</kbd>. All lines of
  that speaker change.
- **Recolour** — pick one of the 10 colours.
- **Merge** — type the name of *another* existing speaker. The app asks
  "Merge…?" and moves all lines to that speaker. Use this when the app found
  **too many** speakers (one person split into two).

## Wrong number of speakers? Re-detect them

If the app found **too few** speakers (two people shown as one), or you set the
wrong number before starting:

1. Open the transcript.
2. Click **👥 Re-detect speakers** (right of the speaker chips).
3. Pick the correct number (or Auto-detect) and click **Re-detect**.

What happens:

- Only speaker detection runs again. The speech-to-text is **not** redone, so it
  takes a fraction of the time (seconds for short files, about 1–2 minutes for an
  hour of audio on a fast machine).
- The result is a **new job** in Recent jobs (marked *re-detected*). The original
  stays, so you can compare and delete the one you don't want.
- Your **text** edits are kept. Speaker **names and colours** are reset, because
  the speakers are new — rename them again.
- It counts as a job for the "keep last N jobs" limit.

Do you need to transcribe again from scratch? Only if the **words** are wrong
(wrong language, wrong model). For speaker problems, Re-detect is enough.

## Simplified ↔ Traditional Chinese

Whisper often writes Cantonese in Simplified characters (欢迎, 会). Two ways to fix it:

- **Before transcribing:** set *Chinese characters* to **Traditional — Hong Kong**
  (or Taiwan, or Simplified). Every new transcript is converted automatically.
- **After transcribing:** open the transcript and use the **Characters → Convert to…**
  menu above it. The change is saved, and all exports use it.

The conversion uses [OpenCC](https://github.com/BYVoid/OpenCC), which converts by
phrase, not letter by letter: 头发 → 頭髮 but 发展 → 發展. The Hong Kong option uses
Hong Kong forms (软件 → 軟件); the Taiwan option also uses Taiwan words
(软件 → 軟體). Cantonese characters such as 我哋, 嘅 and 喺 are left as they are.

## Whisper or Qwen3-ASR?

Both are speech-to-text models; you choose in **Models**.

| | Whisper large-v3 | Qwen3-ASR 1.7B / 0.6B |
|---|---|---|
| Cantonese accuracy | Good; often writes standard Chinese instead of spoken Cantonese | **Much better** (published error rate about half of Whisper's); keeps 我哋 / 嘅 / 喺 |
| Mandarin / English | Very good | Mandarin better, English about the same |
| Word timings (karaoke, speaker split) | Yes | Yes (via the Qwen3-ForcedAligner, downloaded with it) |
| Download | 3.1 GB | 6.5 GB (1.7B) or 3.7 GB (0.6B) |
| Memory needed | ~6 GB | ~8 GB (1.7B) or ~4 GB (0.6B) |
| License | Apache-2.0 | Apache-2.0 |

To switch: **Models → Speech-to-text models → Download** next to Qwen3-ASR, wait
for it, then pick it under **Active models** and click **Apply & load**. Switch
back to Whisper the same way. Both can stay downloaded.

## Export (Word, subtitles, text, JSON)

Buttons at the top of the transcript:

| Button | File | Use it for |
|---|---|---|
| **DOCX** | Word document with a table: ID · Start · End · Person · : · Transcript. Fonts: **Times New Roman** for English, **PMingLiU (新細明體)** for Chinese | Reports, printing |
| **SRT** / **VTT** | Subtitle files with `[Speaker]` labels | Video players, YouTube, editors |
| **TXT** | One line per segment: `[Speaker] text` | Pasting into email or chat |
| **JSON** | Everything, including word timings | Other programs |

PMingLiU ships with Windows and Microsoft Office. On a Mac without Office, Word
or Pages may substitute a similar Chinese font — the document still says
PMingLiU, so it shows correctly on machines that have it.

## Dashboard

- **Server / device / models** — what is running, and on which hardware.
- **Share access** — the addresses other computers can use, with copy buttons.
- **Server log** — see below.
- **API endpoints** — for developers ([API reference](API.md)).

## Server log

The **Server log** card on the Dashboard shows the last 300 lines the server
wrote, so you can see what happened without access to the server's terminal.

**Request lines** controls how much is written about web requests:

| Mode | What is logged |
|---|---|
| **Quiet** (default) | Uploads, edits, deletes, cancels and every error. Hides the routine "page refresh" requests (`GET /api/jobs 200`, `GET /api/status 200`) that every open browser tab makes every 10 seconds |
| **Full** | Every request — use this when you debug a client |
| **Off** | No request lines; job and model messages still appear |

The change applies at once, for the terminal and the log file.

- **Clear** empties the view and starts a new log file (the previous one is kept
  as `sst.log.1`).
- **Download** saves the current log file.
- The full log lives on the server at `data/logs/sst.log`. It rotates at 5 MB
  and keeps 3 old files, so it never uses more than about 20 MB.

Browser tabs in the background no longer ask the server for status every few
seconds, which also cuts the noise.

## Models tab

- **Active models** — choose which downloaded speech-to-text and speaker
  model to use, then **Apply & load**. A switch during a running job waits for
  the job to finish.
- **Catalog** — download, cancel a download, or remove models. Each card shows
  the size, languages and license.
- **Search Hugging Face** — find more Whisper models (for example
  `whisper cantonese`). Only standard PyTorch models can run; others show
  "unsupported format".

More detail, licenses and a comparison of newer models: [MODELS.md](MODELS.md).

## Settings tab

| Card | What it does |
|---|---|
| **Hugging Face token** | Needed once to download the gated pyannote speaker models. Create a free token, accept the model terms (links in the card), paste, Save |
| **Compute device** | Auto (recommended), or force CUDA / Apple GPU (MPS) / CPU. Takes effect next time models load |
| **Job history** | How many finished jobs to keep (3–20) |
| **Access & security** | Set an API key and turn on **Require login for other devices**. Other computers must then log in with the key; the server's own computer never has to |
| **Server port** | Change the port; the server restarts on it and your browser follows |

## Use it from another computer

1. On the server, open **Dashboard → Share access** and copy an address, such as
   `http://192.168.1.20:8756` (same Wi-Fi/office network) or the Tailscale
   address `http://100.x.y.z:8756` (works anywhere on your tailnet).
2. Open that address on the other computer.
3. If login is on, enter the API key.

## FAQ

**The cancel button says "Cancelling…" — is it stuck?**
No. It stops at the next safe point, normally within 1–3 seconds. On a very
slow or overloaded machine it can take a little longer.

**I set the speaker count wrong.** Use [Re-detect speakers](#wrong-number-of-speakers-re-detect-them).

**Two different people got the same label.** Re-detect with a higher number,
or fix single lines with the speaker list in edit mode.

**One person got two labels.** Rename one label to the other's name to merge.

**Cantonese comes out in Simplified characters.** Set *Chinese characters* to
Traditional — Hong Kong, or use *Convert to…* on the transcript.

**Cantonese comes out as written standard Chinese instead of spoken Cantonese.**
Set Language to Cantonese, or switch to Qwen3-ASR (see above).

**Model downloads are slow.** Downloads come from Hugging Face, the official home
of these models, with its fast multi-connection downloader. In a speed test,
ModelScope (Alibaba's official mirror, which also hosts Qwen) was no faster than
Hugging Face, because the internet connection was the limit. A download that
stops resumes when you click Download again. To save time on a second machine,
copy the downloaded models instead (see [MODELS.md](MODELS.md#offline--air-gapped-use)).
Avoid unofficial "mirror" sites — they are not run by the model authors.

**Where are my files stored?** In the project's `data/` folder on the server:
`data/audio` (audio), `data/jobs` (transcripts), `data/logs` (logs),
`data/config.json` (settings). Models are in `~/.cache/huggingface`.
