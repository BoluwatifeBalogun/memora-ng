# Notes on the write-up (Chapters 1 to 3)

Things found while reading the document closely. The first group affects the
implementation. The rest are corrections worth making before submission.

## A. Decisions the implementation had to make

1. **Dataset.** Chapter 3 names ADNI. ADNI requires an approved application, so it
   cannot simply be downloaded. The training notebook uses OASIS-1, which is open and
   keeps subject and slice numbers, and it also accepts an ADNI export. Either get
   ADNI access, or change 3.3, 3.5, 3.7 and 3.8 to name OASIS-1 (Marcus et al., 2007).
2. **Classes.** 3.4 and 3.5 say Cognitively Normal, MCI and Alzheimer's disease. The
   "Expected Contributions" in Chapter 2 list four stages (Non-Demented, Very Mild,
   Mild, Moderate). The system follows Chapter 3 with three classes. On OASIS-1 they map
   to CDR 0, CDR 0.5 and CDR 1 or 2. Make the Chapter 2 list match.
3. **One image or many.** The text says the user uploads "an MRI brain image", but an
   LSTM needs a sequence. The system takes a series of slices or one NIfTI volume, and
   accepts a single slice with a "limited context" warning. Reword 3.12.2, 3.12.4 and
   3.12.14 to "an MRI slice series".
4. **Hypothesis.** H1 says the hybrid "significantly improves" detection. That needs a
   comparison. `train.py --vit-only` trains the same ViT without the LSTM on the same
   split so Chapter 4 can report both.
5. **Framework.** The text lists TensorFlow and PyTorch. The model is written in
   PyTorch and served with ONNX Runtime. Say so in 1.9, 3.9 and 3.12.12.
6. **Deployment diagram (3.12.9).** API Gateway, Authorizer, Content Service, User
   Service and External API are not parts of this system. The real deployment is:
   browser, Gunicorn and Flask, ONNX Runtime, MySQL, file storage.
7. **Class diagram (3.12.6).** "Database" is not a domain class. The Patient class
   lists MRI Image Reference and Date of Examination, which belong to MRI_Images.

## B. Text and structure

8. The document title says "LTMS". It should be "LSTM".
9. 1.2, first paragraph: stray words "delayed diagnosis" after "(WHO, 2023)".
10. 2.2: the placeholder "Example flow:" was left in.
11. 2.4.1 is headed "CNN-Based ... Studies" but all five studies are about Vision
    Transformers. No CNN study is reviewed, although Jain et al. (2019) is in the
    reference list. Add two or three CNN studies or rename the section.
12. Studies Four and Five have no authors ("published in BMC Neurology (2023)", "published
    in 2025"). Add the authors and full references.
13. 2.4.3 and 2.4.4 contain no citations at all, and the "Synthesis of Key Hybrid
    Studies" table names no studies.
14. 2.5.4 refers to "Table 2.1" but the table is titled "Critical Research Gap Matrix".
    Number all tables and figures consistently (figures jump between 2.2, 2.5.1, 3.12.1...).
15. 3.12.1 ends with "follow the sequence below:" and nothing follows.
16. 3.12.8: the heading "Users Table" sits above a table that covers all four tables.
17. Figure sources such as "dzm1try", "Tomas Kn0pp", "Andresr", "Islander11" and
    "VectorMine" are stock-image contributors. Use your own diagrams of this system.

## C. Citations

18. Several citations are attached to statements the source does not make:
    Dosovitskiy et al. (2021) for MRI biomarkers (1.1); Hochreiter and Schmidhuber (1997)
    for inter-observer variability (1.1); Goodfellow et al. (2016) and Li et al. (2023)
    for early symptoms (1.1); Jack et al. (2008) for treatment timing and for reading
    time (1.1, 1.7); Pressman and Maxim (2020) for image preprocessing (1.9);
    Sommerville (2016) for evaluation metrics (1.9).
19. Cited in the text but missing from the references: WHO (2023), Alzheimer's
    Association (2024), Russell and Norvig (2021), Snyder (2019), Odusami et al. (2023),
    Hoang et al. (2023), Hatamizadeh et al. (2022), Shamshad et al. (2023), He et al.
    (2023), Khan et al. (2024).
20. In the references but never cited: Alp et al. (2025), Jain et al. (2019),
    Stallings (2018), Vaswani et al. (2017).
21. "Stallings, W. (2018). Database systems: Design, implementation, and management"
    is attributed to the wrong author. That Cengage title is by Coronel and Morris.
22. Citation style is mixed: "(Author, Year)" and "Author (Year)" both appear at the
    ends of sentences. Use one form.
