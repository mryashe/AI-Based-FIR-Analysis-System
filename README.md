<img width="8192" height="5464" alt="pexels-dpsinghbhullar-30483132" src="https://github.com/user-attachments/assets/d1548c74-b61b-4593-9ac2-f985d1b7680e" />
⚖️ AI-Based FIR Analysis System for Automatic Identification of Applicable BNS Sections

An AI/NLP web app that reads a crime description in plain language and instantly suggests the relevant **Bharatiya Nyaya Sanhita (BNS)** sections, with a simple explanation, the punishment, and the reason each section was suggested.

## ❗ Problem

At FIR registration, the applicable BNS sections are not always known immediately. Identifying them needs legal expertise, which leads to delays, human error, and sometimes wrong or missing sections that can weaken the case in court.

## 🎯 Objectives

- Understand crime descriptions written in everyday language
- Automatically match them to the correct BNS sections
- Reduce dependence on manual legal knowledge at the FIR stage
- Make FIR filing faster, more accurate, and more transparent

## ✨ Features

- Plain-language text (or voice) input
- Ranked BNS section suggestions with title, explanation, and punishment
- Explainable AI (XAI): shows *why* a section was suggested
- Optional: Hindi/Marathi/English support and FIR draft generator

## ⚙️ How It Works

1. User enters the incident description
2. Text is cleaned and preprocessed (NLP)
3. Model detects crime type and keywords
4. Description is matched against the BNS database
5. Sections are ranked and shown with explanations
6. Optional FIR draft is generated

## 🧰 Tech Stack

| Area | Technology |
|---|---|
| Language | Python |
| AI / NLP | ML / DL, text classification, NER, embeddings, optional LLM |
| Frontend | HTML, CSS, JavaScript  |
| Backend | Django  |
| Database | MySQL / MongoDB |
| Explainability | XAI methods |


## 🧪 Example

**Input:** "A person broke into my house at night and stole jewellery."

**Output:** ranked BNS sections (e.g., theft, house-breaking by night) with explanations and punishments.

## ⚠️ Disclaimer

Suggestions only, not legal advice. Final section selection must be made by a qualified police officer or legal professional.

## 🔭 Future Scope

- Extend to BNSS and BSA
- Citizen chatbot and voice-to-text
- Integration with police record-management systems
- Similar past-case recommendations



