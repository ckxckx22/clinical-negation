import pandas as pd
import random


def to_df(data_dir, column_names=None, trim_col=[]):
    with open(data_dir, "r") as f:
        lines = f.readlines()
        lines = [line.split('\t') for line in lines]
    if not column_names:
        column_names = ['token', 'begin', 'end', 'section', 'filename', 'concept', 'label']
    df = pd.DataFrame(lines, columns = column_names)
    if trim_col:
        for col in trim_col:
            df[col] = df[col].str.rstrip("\n")
    return df


def write(df):
    def concat_row_values(row):
        if row.iloc[1] is None:
            return ""
        return '\t'.join(row.values.astype(str))

    # Apply the function across rows and join the resulting strings with new lines
    final_text = df.apply(concat_row_values, axis=1).str.cat(sep='\n')

    return final_text


if __name__ == "__main__":
    which_set = "train"
    DATA_DIR = "/Users/chenkx/git/clinical-negation/emnlp2017-bilstm-cnn-crf/data/i2b2_2010/real/%s.txt" % which_set
    highly_negated = ["Physical examination/Status", "Review of systems", "Allergies", "Complications"]
    # lowly_negated = ["Patient information/Demographics", "Present illness", "Hospital course", "Social history",
    #                  "Family history", "Addendum", "Radiology", "Unknown/Unclassified", "Problems",
    #                  "Reasons/Indications",
    #                  "Procedures/Surgery", "Chief complaint", "Nutrition", "Past history", "Assessment", "Diagnoses",
    #                  "Laboratory tests", "Follow-up/Instructions", "Assessment/Plan", "Allergies", "Medications",
    #                  "Investigations/Results"]

    raw = to_df(DATA_DIR, trim_col=["label"])
    raw["row_id"] = raw.index.to_list()
    raw['section'] = raw['section'].fillna(method='ffill')
    low_subset = raw[~raw.section.isin(highly_negated)]
    high_subset = raw[raw.section.isin(highly_negated)]

    # # write to file: the subset of highly-negated sections
    # with open("/Users/chenkx/git/clinical-negation/emnlp2017-bilstm-cnn-crf/data/i2b2_2010_highly_negated/%s.txt" % which_set, "w") as f:
    #     f.write(write(high_subset))
    # # write to file: the subset of the rest of the dataset
    # with open("/Users/chenkx/git/clinical-negation/emnlp2017-bilstm-cnn-crf/data/i2b2_2010_lowly_negated/%s.txt" % which_set, "w") as f:
    #     f.write(write(low_subset))

    # down-sample the lowly_negated subset to 10%
    random.seed(10)
    to_keep = random.sample(low_subset.row_id[low_subset.begin.isna()].to_list(), sum(high_subset.begin.isna()))
    downsampled = low_subset.copy()
    # downsampled = raw.copy()
    downsampled["keep"] = None
    downsampled.keep[(downsampled.begin.isna()) & (downsampled.row_id.isin(to_keep))] = True
    downsampled.keep[(downsampled.begin.isna()) & (~downsampled.row_id.isin(to_keep))] = False
    downsampled.keep = downsampled.keep.fillna(method="bfill")
    print("Ratio of the number of sentences in the subsets of the highly-negated sections and the low_subset set: %.4f(%d/%d)"
          % (sum(high_subset.begin.isna())/sum(low_subset.begin.isna()),
             sum(high_subset.begin.isna()), sum(low_subset.begin.isna())))
    print("After downsampling, ratio of the number of tokens in the highly-negated subset and the low_subset set: %.4f(%d/%d)"
          % (downsampled.keep.sum() / len(high_subset), downsampled.keep.sum(), len(high_subset)))

    downsampled = downsampled[downsampled.keep]

    # with open("/Users/chenkx/git/clinical-negation/emnlp2017-bilstm-cnn-crf/data/i2b2_2010_downsample-lowly_negated/%s.txt" % which_set, "w") as f:
    #     f.write(write(downsampled))
    # print(downsampled)



