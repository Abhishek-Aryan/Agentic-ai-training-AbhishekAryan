🤖 AI Short-Form Video Generation & Auto-Publishing Pipeline
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Powered by n8n, OpenRouter, PIAPI, Createmate & Blotato
--
A fully automated end-to-end system that generates AI battle shorts (like Ape vs Lion) from scratch — complete with scenes, close-ups, aftermath images, video rendering, and multi-platform publishing — all for under $1 per video.

🎬 Overview
-----
This project automates the entire short-form content creation pipeline using AI agents, automation workflows, and low-cost APIs.
The workflow is designed and executed in n8n, and uses Google Sheets for content sourcing, AI models for prompt & image generation, Createmate for rendering, and Blotato for auto-publishing.


<img width="592" height="593" alt="{32B76953-80CB-445D-A0C5-FE6F9489BF57}" src="https://github.com/user-attachments/assets/985c87f2-719c-498e-96c8-e9fd3a31a065" />

🌟 Key Highlights
-----
Automatically generates fight matchups, scenes, and visuals

Uses Flux image model via PIAPI for realistic imagery

Fully template-based video rendering in Createmate

Auto-posts to YouTube Shorts, TikTok, and Instagram

Cost per full video: Under $1

100% reproducible via free workflow JSON and Google Sheet template

🧩 Workflow Architecture
🟩 1. Create Scenes
--
Generates the main concept and matchups.

Steps:
-
.Triggered from Google Sheet (e.g., “Ape vs Big Cats”).

.AI agent (OpenRouter → Chat model) generates 8 matchups like:

.Ape vs Lion

.Ape vs Tiger

.Ape vs Leopard

.Ape vs Jaguar
(and so on)

.Output: Scene JSON with structured data (main character + opponents).
<img width="792" height="278" alt="{F6456410-5420-4757-8C22-4FDFF1962C00}" src="https://github.com/user-attachments/assets/5539c0b9-13df-4cc0-a1c3-5988e1a6c9f0" />


🟦 2. Create Close-up Images
---
Generates 16 square close-up images (two per matchup) via Flux/PIAPI.

Steps:
-
.Prompts are generated dynamically from scene text.

.Images are generated using PIAPI (POST https://api.piapi.ai/generate).

.Waits 90 seconds to ensure completion.

.Aggregates all generated images and updates Google Sheets.

.Output: 16 intimidating close-up images ready for video.
<img width="795" height="202" alt="{13B53322-FA84-4A35-8BBC-B0C178853CF0}" src="https://github.com/user-attachments/assets/71caf41f-8c8f-4229-bb83-09c2efe42a7f" />


🟪 3. Create Winner Images
--
Determines winners and generates portrait-style aftermath visuals.

Steps:
-
.Second AI agent (OpenRouter → Chat model) evaluates likely winners.

.Generates 8 winner prompts (e.g., “lion stands victorious over ape”).

.PIAPI generates the final set of portrait aftermath images.

.Data is aggregated and synced with Sheets.

.Output: 8 realistic winner/aftermath images.
<img width="799" height="198" alt="{426929EB-2B0B-4C0F-8CC5-7D4C9F1528CF}" src="https://github.com/user-attachments/assets/59b0a2e6-1378-42f0-905a-670268bde467" />


🟥 4. Render Video
-
All generated images are compiled into a short cinematic video using Createmate.

Steps:
-
.Fetches all image URLs from Google Sheets.

.Uses a predefined Createmate template for layout and transitions.

.Sends render job via API (POST https://api.createmate.ai/render).

.Waits 90 seconds for video generation.

.Saves final video link to Google Sheets.

.Output: 1 complete, high-quality short video.
<img width="807" height="199" alt="{DFB43D2A-BE7D-4EB9-9B09-CE68AA11D6BF}" src="https://github.com/user-attachments/assets/65d9f244-f894-4f9c-9ee8-ba570c21254e" />

🟧 5. Auto-Publish to Platforms
--
Uploads the generated video to YouTube Shorts, TikTok, and Instagram automatically via Blotato.

Steps:
-
.Blotato API handles uploads and scheduling.

.Video titles, descriptions, and tags are dynamically pulled from Google Sheets.

.Output: Multi-platform published video.
<img width="263" height="483" alt="{DE8C0840-880C-42C1-B327-DFDE300126A6}" src="https://github.com/user-attachments/assets/17dbd86d-f32d-4a71-b27e-53c147997146" />

⚙️ Setup Guide
---
.Make a copy of this Google Sheet Template

.Connect it to all 5 Google Sheet nodes inside n8n.

.Connect your OpenRouter API key

.Used in the two Chat Model nodes for scene and winner generation.

.Create a PIAPI account
 
.Get your API key and add it to the image generation nodes.

.Create a Createmate account

.Connect your Template ID and Account ID.

.Duplicate the same video template shown in the reference video using the same Skoop point.

.Connect your Blotato account

.Add your API key to enable auto-publishing to Instagram, TikTok, and YouTube.


🚀 Quick Start
---
<img width="707" height="616" alt="{63A579B3-4928-495B-9255-AEF1CCFD9721}" src="https://github.com/user-attachments/assets/53e4365c-b439-4777-b424-7f71f824c4ed" />

.Import the Workflow

.In n8n → “Import Workflow” → Upload ai_video_automation.json

.Connect Your Credentials

.Google Sheets, OpenRouter, PIAPI, Createmate, Blotato

.Customize Template

.Update your own Createmate template and test rendering.

.Run the Workflow

.Trigger manually or schedule it with n8n’s built-in Scheduler node.

.Check Results

.Generated assets and video URLs appear in Google Sheets.


📊 Example Output
--
Prompt Example: “Ape vs Lion, intense jungle battle at dawn.”

Generated Assets:

16 Close-up Images

8 Winner Portraits

1 Rendered Video

Auto Uploaded To:
--
✅ YouTube Shorts

✅ TikTok

✅ Instagram

🧩 Future Improvements
---
🔊 Add AI voiceovers using ElevenLabs or OpenAI TTS

🈵 Multi-language prompt generation

✍️ Dynamic captions & subtitles via Whisper or Gemini

📅 Scheduled content calendars synced to Sheets
