// Power Query (M) — Legal AI (RAG vs fine-tuning) per-question results
//
// Usage: Power BI Desktop > Get Data > Blank Query > Advanced Editor > paste this,
// then set FilePath below to the local path of results/difficulty_table.csv.
//
// This is the real per-question evaluation table (222 test questions), with
// difficulty features (question length, number of gold articles required,
// category training frequency) — the source for the difficulty/quality
// correlation analysis described in RESULTS.md.

let
    FilePath = "C:\Path\To\legal-ai\results\difficulty_table.csv",

    Source = Csv.Document(
        File.Contents(FilePath),
        [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]
    ),
    PromotedHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),

    Typed = Table.TransformColumnTypes(PromotedHeaders, {
        {"id", Int64.Type},
        {"category", type text},
        {"subcategory", type text},
        {"question", type text},
        {"extra_description", type text},
        {"article_ids", type text},
        {"question_length_words", Int64.Type},
        {"n_gold_articles", Int64.Type},
        {"category_train_frequency", type number},
        {"difficulty_bucket", type text}
    }),

    // article_ids arrives as a Python-list-looking string, e.g. "[947, 948]" —
    // turn it into an actual list of numbers so it can be expanded to one row per cited article
    AddArticleList = Table.AddColumn(Typed, "article_id_list", each
        let
            cleaned = Text.Replace(Text.Replace(Text.Replace([article_ids], "[", ""), "]", ""), " ", ""),
            parts = if cleaned = "" then {{}} else Text.Split(cleaned, ",")
        in
            List.Transform(parts, each Number.FromText(_)),
        type list
    ),

    Final = Table.RemoveColumns(AddArticleList, {"article_ids"})

    /* To explode one row per cited article (e.g. for a "citations per article" report),
       replace the line above with:
       Final = Table.ExpandListColumn(AddArticleList, "article_id_list")
    */
in
    Final
