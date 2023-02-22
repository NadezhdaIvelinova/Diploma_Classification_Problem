using Microsoft.ML;
using Microsoft.ML.Transforms.Text;
using System;
using System.Collections.Generic;
using System.Text;

namespace ClassificationModel_ConsoleApp
{
    class Program
    {
        static void Main(string[] args)
        {
            var input = Console.ReadLine();
            // Create single instance of sample data from first line of dataset for model input
            ClassificationModel.ModelInput sampleData = new ClassificationModel.ModelInput()
            {
                //Col0 = @"Documentation seems simple, but following them does not really work. UI is slow.",
                Col0 = input
            };

            // Make a single prediction on the sample data and print results
            var predictionResult = ClassificationModel.Predict(sampleData);

            
            float errorScore = (float)predictionResult.Score.GetValue(0);
            float feedbackScore = (float)predictionResult.Score.GetValue(1);

            if (predictionResult.Prediction == 0)
            {
                 
                Console.WriteLine($"\n\nPredicted: Feedback\nError/Action Needed Score: {Math.Round(errorScore, 2)} \nFeedback Score {Math.Round(feedbackScore, 2)}\n\n");
            }
            else
            {
                Console.WriteLine($"\n\nPredicted: Error\nError/Action Needed Score: {Math.Round(errorScore, 2)} \nFeedback Score {Math.Round(feedbackScore, 2)}\n\n");
            }

            //Tokenize anc clear stop words
            if (predictionResult.Prediction == 1)
            {
                var textData = new TextData { Text = input };
                Tokenizer(textData);
                ClearStopWords(textData);
            }
            else
            {

            }

            Console.ReadKey();
        }

        private static void PrintTokens(TextTokens tokens)
        {
            Console.Write(Environment.NewLine);

            var sb = new StringBuilder();

            foreach (var token in tokens.Tokens)
            {
                sb.AppendLine(token);
            }

            Console.WriteLine(sb.ToString());
        }

        private static void Tokenizer(TextData textData)
        {
           
            var context = new MLContext();
            var emptyData = new List<TextData>();
            var data = context.Data.LoadFromEnumerable(emptyData);

            var tokenization = context.Transforms.Text.TokenizeIntoWords("Tokens", "Text", separators: new[] { ' ', ',', '.' });
            var tokenModel = tokenization.Fit(data);
            var engine = context.Model.CreatePredictionEngine<TextData, TextTokens>(tokenModel);
            var tokens = engine.Predict(textData);

            PrintTokens(tokens);
        }

        private static void ClearStopWords(TextData textData)
        {
            var mlContext = new MLContext();
            var emptySamples = new List<TextData>();

            // Convert sample list to an empty IDataView.
            var emptyDataView = mlContext.Data.LoadFromEnumerable(emptySamples);

            var textPipeline = mlContext.Transforms.Text.TokenizeIntoWords("Words",
                "Text")
                .Append(mlContext.Transforms.Text.RemoveDefaultStopWords(
                "WordsWithoutStopWords", "Words", language:
                StopWordsRemovingEstimator.Language.English));

            // Fit to data.
            var textTransformer = textPipeline.Fit(emptyDataView);

            // Create the prediction engine to remove the stop words from the input
            // text /string.
            var predictionEngine = mlContext.Model.CreatePredictionEngine<TextData,
                TransformedTextData>(textTransformer);

            // Call the prediction API to remove stop words.

            var prediction = predictionEngine.Predict(textData);

            // Print the length of the word vector after the stop words removed.
            Console.WriteLine("Number of words: " + prediction.WordsWithoutStopWords
                .Length);

            // Print the word vector without stop words.
            Console.WriteLine("\nKeywords: " + string.Join(",",
                prediction.WordsWithoutStopWords));
        }
    }
}
