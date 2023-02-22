using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ClassificationModel_ConsoleApp
{
    public class TransformedTextData : TextData
    {
        public string[] WordsWithoutStopWords { get; set; }
    }
}
