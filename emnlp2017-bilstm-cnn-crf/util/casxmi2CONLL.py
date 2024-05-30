## Convert files from CAS XMI annotated format to CoNLL format
import os

import argparse
import glob

import cassis

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input-dir",
    dest="input_dir",
    type=str,
    default='',
    help="Input directory where CAS XMI annotations are stored",
)

parser.add_argument(
    "--output-file",
    dest="output_file",
    type=str,
    default='',
    help="Output file where CoNLL format annotations are saved",
)

class FormatConvertor:
    
    def __init__(self, input_dir: str, output_file: str = "", 
                 type_system_loc='/Users/chenkx/data/GCS/TypeSystem.xml',
                 # GCS phrase type: ExplicitGCSMention, ResponseRelatedToGCS, BarrierToGCSAssessment, AlternativeScale
                 gcs_phrase_type='all'):
        self.input_dir = input_dir
        self.output_file = output_file
        ## TODO - make the file suffix configurable
        self.file_suffix = '.xmi'
        self.type_system = None
        with open( type_system_loc , 'rb' ) as fp:
            self.type_system = cassis.load_typesystem( fp )
        ## TODO - make sentence and token types configurable
        self.sentence_type = 'de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence'
        self.token_type = 'de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token'
        ## TODO - make entity type configurable
        self.entity_type = 'webanno.custom.GCSAnnotationLayer'
        assert gcs_phrase_type in ["all",
                                   "ExplicitGCSMention", 
                                   "ResponseRelatedToGCS", 
                                   "BarrierToGCSAssessment", 
                                   "AlternativeScale"]
        self.gcs_phrase_type = gcs_phrase_type
        

    def parse_text_for_one_file(self, this_filename):
        with open( os.path.join( self.input_dir , this_filename ) , 'rb' ) as fp:
            ##print( '{}'.format( this_filename ) )
            cas = cassis.load_cas_from_xmi( fp ,
                                            typesystem = self.type_system )
        #for t in cas.typesystem.get_types():
        #    print( '{}'.format( t ) )

        output_lines = []

        covered_spans = { 'ExplicitGCSMention' : {} ,
                            'ResponseRelatedToGCS' : {} ,
                            'BarrierToGCSAssessment' : {} ,
                            'AlternativeScale' : {} }
        for entity in cas.select( self.entity_type ):
            phrase_type = entity[ 'PhraseType' ]
            begin_pos = entity[ 'begin' ]
            end_pos = entity[ 'end' ]
            ##
            concept_value = 'Unknown'
            if( phrase_type == 'ExplicitGCSMention' ):
                1 ## TODO - find examples of Explicit GCS Mention
            elif( phrase_type == 'ResponseRelatedToGCS' ):
                1 ## TODO - find examples of ResponseRelatedToGCS
            elif( phrase_type == 'BarrierToGCSAssessment' ):
                concept_value = entity[ 'PotentialAssessmentBarrier' ]
            elif( phrase_type == 'AlternativeScale' ):
                concept_value = entity[ 'AlternativeScaleType' ]
            ##
            if( begin_pos in covered_spans[ phrase_type ] ):
                ## TODO - handle (and report more verbosely) collisions
                print( 'Error:  collision' )
            else:
                covered_spans[ phrase_type ][ begin_pos ] = 'B-{}'.format( concept_value )
            ## Check if this annotation spans more than one character
            if( begin_pos + 1 < end_pos ):
                for i in range( begin_pos + 1 , end_pos ):
                    if( i in covered_spans[ phrase_type ] ):
                        print( 'Error:  collision' )
                    else:
                        covered_spans[ phrase_type ][ i ] = 'O-{}'.format( concept_value )
        ##print( '{}'.format( covered_spans ) )
        sentence_count = 0
        for sentence in cas.select( self.sentence_type ):
            token_count = 0
            for token in cas.select_covered( self.token_type , sentence ):
                gcs_feature = {'ExplicitGCSMention': 'O', 
                                'ResponseRelatedToGCS': 'O', 
                                'BarrierToGCSAssessment': 'O', 
                                'AlternativeScale': 'O'}
                if( token.begin in covered_spans[ 'ExplicitGCSMention' ] ):
                    gcs_feature['ExplicitGCSMention'] = covered_spans[ 'ExplicitGCSMention' ][ token.begin ]
                if( token.begin in covered_spans[ 'ResponseRelatedToGCS' ] ):
                    gcs_feature['ResponseRelatedToGCS'] = covered_spans[ 'ResponseRelatedToGCS' ][ token.begin ]
                if( token.begin in covered_spans[ 'BarrierToGCSAssessment' ] ):
                    gcs_feature['BarrierToGCSAssessment'] = covered_spans[ 'BarrierToGCSAssessment' ][ token.begin ]
                if( token.begin in covered_spans[ 'AlternativeScale' ] ):
                    gcs_feature['AlternativeScale'] = covered_spans[ 'AlternativeScale' ][ token.begin ]
                ####
                if self.gcs_phrase_type == "all":
                    output_lines.append( '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format( token.get_covered_text() ,
                                                                                            token.begin ,
                                                                                            token.end ,
                                                                                            token_count ,
                                                                                            sentence_count ,
                                                                                            this_filename ,
                                                                                            gcs_feature['ExplicitGCSMention'] ,
                                                                                            gcs_feature['ResponseRelatedToGCS'] ,
                                                                                            gcs_feature['BarrierToGCSAssessment'] ,
                                                                                            gcs_feature['AlternativeScale'] ) )
                else:
                    gcs = gcs_feature[self.gcs_phrase_type]
                    if len(gcs.split('-')) < 1:
                        print("GCS: " + gcs)
                    output_lines.append( '{}\t{}\t{}\t{}\t{}\t{}'.format( token.get_covered_text() ,
                                                                            token.begin ,
                                                                            token.end ,
                                                                            "N/A" , 
                                                                            this_filename ,
                                                                           gcs.split('-')[0] ) )

                    
                token_count += 1
            sentence_count += 1
            output_lines.append( '' )
            
        return output_lines


    def parse_text(self, verbose=True) -> str:
        """Loop over all annotation files, and write tokens with their label to an output file"""
        file_list = self.read_input_folder()
        ## TODO - different tokenization options
        ## TODO - different sentence splitting options
        ## TODO - separate folder for preprocessing
        outputs = []
        for this_filename in tqdm( sorted( file_list ) ):
            ##
            output = '\n'.join(self.parse_text_for_one_file( this_filename ))
            outputs.append(output)
            if verbose:
                print( output )
        return '\n'.join(outputs)
        
                
    def read_input_folder(self):
        """Read multiple annotation files from a given input folder"""
        file_list = set( [ os.path.basename(x) for x in glob.glob( os.path.join( self.input_dir ,
                                                                                 '*{}'.format( self.file_suffix ) ) ) ] )
        ##########################
        return file_list

    
if __name__ == '__main__':
    args = parser.parse_args()
    format_convertor = FormatConvertor( args.input_dir , args.output_file )
    format_convertor.parse_text()
