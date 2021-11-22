from argparse import ArgumentParser
from pathlib import Path
from sources.classificators import Models


if __name__ == "__main__":
    parser = ArgumentParser(description='Text theme classification. Use -h for more informations.')

    parser.add_argument('-r', '--rerun', action="store_true", help="Regenerate the classifiers")
    parser.add_argument('-rt', '--train_set', help="Directory with train set", required=False)
    parser.add_argument('-rm', '--rerun_models_dir', help="Directory to save new models", required=False)

    parser.add_argument('-c', '--classify', action="store_true", help="Classify text given in directory")
    parser.add_argument('-ct', '--text_to_classify', help="Directory with text to classify", required=False)
    parser.add_argument('-cm', '--models_dir', help="Directory with models which should be used to classify new texts",
                        required=False)
    args = parser.parse_args()

    print("*********** PARSER INFO **************")
    if args.classify:
        print("Directory with texts to classification: ", args.text_to_classify)
        print("Directory with models which should be used to classification : ", args.models_dir)
    if args.rerun:
        print("Directory with train set: ", args.train_set)
        print("Directory where new models should be saved: ", args.rerun_models_dir)
    if not args.classify and not args.rerun:
        print("No options selected. Use -c or -r. For help use -h")


    if args.classify or args.rerun:

        if args.rerun:
            print(" ++++ RUNNING REGENERATION OF MODELS ++++ \n")
            train_data_dir = str(args.train_set).replace("'", "")
            train_data_dir = train_data_dir.replace('"', '')
            out_models_dir = str(args.rerun_models_dir).replace("'", "")
            out_models_dir = out_models_dir.replace('"', '')
            Path(out_models_dir).mkdir(parents=True, exist_ok=True)

            models_learn = Models(train_data_dir, out_models_dir, None, True)
            models_learn.learnWithBOW(out_models_dir)
            models_learn.learnWithTFIDF(out_models_dir)
            models_learn.learnWithNgram(out_models_dir)
            print("\n\n ++++ LEARNING COMPLETED ++++ ")

        if args.classify:
            print(" ++++ RUNNING TEXT CLASSIFICATION ++++ \n")
            in_text_dir = str(args.text_to_classify).replace("'", "")
            in_text_dir = in_text_dir.replace('"', '')
            models_dir = str(args.models_dir)
            models_dir = models_dir.replace('"', '')

            models_classify = Models(in_text_dir, None, models_dir, False)
            models_classify.classifyUsingExistingModels(models_dir)
            print("\n\n ++++ CLASSIFICATION COMPLETED ++++")