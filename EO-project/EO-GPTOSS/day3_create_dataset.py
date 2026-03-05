"""
Create Training Dataset for Multispectral VLM
Day 3 Afternoon - Dataset Preparation

Generate spectral-aware image-caption pairs
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

class SpectralDatasetCreator:
    """
    Create training dataset with spectral-aware annotations
    
    Strategy:
    1. Use existing 50 synthetic multispectral samples
    2. Generate spectral-aware captions
    3. Create VQA pairs mentioning spectral features
    4. Augment to reach 150-200 samples
    """
    
    def __init__(self):
        self.data_dir = Path('data/raw/sentinel2_multispectral')
        self.output_dir = Path('data/training')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing metadata
        with open(self.data_dir / 'metadata.json', 'r') as f:
            self.samples = json.load(f)
    
    def generate_spectral_captions(self, sample):
        """
        Generate spectral-aware captions for a sample
        
        Uses spectral indices and land cover type to create
        detailed, informative captions
        """
        land_cover = sample.get('land_cover', 'unknown')
        indices = sample.get('spectral_indices', {})
        
        ndvi = indices.get('NDVI', 0)
        ndwi = indices.get('NDWI', 0)
        ndbi = indices.get('NDBI', 0)
        
        captions = []
        
        # Caption templates based on land cover
        if land_cover == 'forest':
            captions.append(
                f"Dense forest area with high vegetation index (NDVI: {ndvi:.2f}), "
                f"showing strong near-infrared reflectance characteristic of healthy vegetation"
            )
            captions.append(
                f"Forested region displaying typical spectral signature with high NIR "
                f"and low red band reflectance, indicating active photosynthesis"
            )
            captions.append(
                f"Natural forest with NDVI of {ndvi:.2f} and negative NDWI ({ndwi:.2f}), "
                f"confirming terrestrial vegetation rather than water"
            )
        
        elif land_cover == 'water':
            captions.append(
                f"Water body with very low near-infrared reflectance and positive NDWI "
                f"({ndwi:.2f}), characteristic of open water surfaces"
            )
            captions.append(
                f"Aquatic area showing strong SWIR absorption and NDWI of {ndwi:.2f}, "
                f"typical of lakes or rivers"
            )
            captions.append(
                f"Water surface with minimal vegetation (NDVI: {ndvi:.2f}) and high "
                f"water index, indicating clear water body"
            )
        
        elif land_cover == 'urban':
            captions.append(
                f"Urban or built-up area with positive NDBI ({ndbi:.2f}) and moderate "
                f"SWIR reflectance, typical of concrete and asphalt surfaces"
            )
            captions.append(
                f"Developed area showing minimal vegetation (NDVI: {ndvi:.2f}) and "
                f"spectral characteristics of built infrastructure"
            )
            captions.append(
                f"Urban landscape with high SWIR values and low vegetation index, "
                f"indicating predominant artificial surfaces"
            )
        
        elif land_cover == 'agriculture':
            captions.append(
                f"Agricultural cropland with high NDVI ({ndvi:.2f}) showing active "
                f"vegetation and strong red-edge response typical of healthy crops"
            )
            captions.append(
                f"Cultivated fields displaying high near-infrared reflectance and "
                f"NDVI of {ndvi:.2f}, indicating productive agricultural land"
            )
            captions.append(
                f"Farmland with vegetation index of {ndvi:.2f} and spectral signature "
                f"consistent with growing crops in good condition"
            )
        
        elif land_cover == 'bare_soil':
            captions.append(
                f"Bare soil surface with low NDVI ({ndvi:.2f}) and high SWIR reflectance, "
                f"showing minimal vegetation cover"
            )
            captions.append(
                f"Exposed ground with vegetation index near zero and spectral characteristics "
                f"typical of dry, unvegetated soil"
            )
            captions.append(
                f"Non-vegetated terrain with NDVI of {ndvi:.2f} and positive NDBI, "
                f"indicating bare earth or sparse vegetation"
            )
        
        return captions
    
    def generate_vqa_pairs(self, sample):
        """Generate VQA question-answer pairs"""
        land_cover = sample.get('land_cover', 'unknown')
        indices = sample.get('spectral_indices', {})
        
        ndvi = indices.get('NDVI', 0)
        ndwi = indices.get('NDWI', 0)
        ndbi = indices.get('NDBI', 0)
        
        qa_pairs = []
        
        # Common questions
        qa_pairs.append({
            'question': 'What type of land cover is visible in this image?',
            'answer': f'{land_cover.replace("_", " ").title()}'
        })
        
        qa_pairs.append({
            'question': 'What is the NDVI value?',
            'answer': f'The NDVI is approximately {ndvi:.2f}'
        })
        
        # Land-cover specific questions
        if land_cover == 'forest':
            qa_pairs.extend([
                {
                    'question': 'Is there vegetation present?',
                    'answer': 'Yes, there is dense vegetation with high near-infrared reflectance'
                },
                {
                    'question': 'What spectral bands indicate vegetation?',
                    'answer': 'High NIR (Band 8) and low Red (Band 4) reflectance indicate healthy vegetation'
                },
                {
                    'question': 'Is the vegetation healthy?',
                    'answer': f'Yes, the NDVI of {ndvi:.2f} indicates healthy, photosynthetically active vegetation'
                }
            ])
        
        elif land_cover == 'water':
            qa_pairs.extend([
                {
                    'question': 'Are there water bodies visible?',
                    'answer': 'Yes, the area shows water with characteristic low NIR and high NDWI'
                },
                {
                    'question': 'What is the NDWI value?',
                    'answer': f'The NDWI is {ndwi:.2f}, confirming water presence'
                },
                {
                    'question': 'What spectral signature indicates water?',
                    'answer': 'Very low NIR and SWIR reflectance are diagnostic for water bodies'
                }
            ])
        
        elif land_cover == 'urban':
            qa_pairs.extend([
                {
                    'question': 'Is this area urban or rural?',
                    'answer': 'Urban, with built-up surfaces showing high SWIR and positive NDBI'
                },
                {
                    'question': 'What is the NDBI value?',
                    'answer': f'The NDBI is {ndbi:.2f}, indicating built-up area'
                },
                {
                    'question': 'Is there vegetation present?',
                    'answer': f'Minimal vegetation, with NDVI of only {ndvi:.2f}'
                }
            ])
        
        elif land_cover == 'agriculture':
            qa_pairs.extend([
                {
                    'question': 'What type of agriculture is visible?',
                    'answer': 'Active cropland with high vegetation index indicating growing crops'
                },
                {
                    'question': 'Are the crops healthy?',
                    'answer': f'Yes, NDVI of {ndvi:.2f} indicates healthy, vigorous crop growth'
                },
                {
                    'question': 'What spectral features indicate crops?',
                    'answer': 'High NIR reflectance and strong red-edge response are typical of healthy crops'
                }
            ])
        
        elif land_cover == 'bare_soil':
            qa_pairs.extend([
                {
                    'question': 'Is there vegetation present?',
                    'answer': f'No or minimal vegetation, NDVI is only {ndvi:.2f}'
                },
                {
                    'question': 'What type of surface is this?',
                    'answer': 'Bare soil or exposed ground with minimal vegetation cover'
                },
                {
                    'question': 'What spectral signature indicates bare soil?',
                    'answer': 'Low NDVI, moderate SWIR reflectance, and lack of NIR-Red contrast'
                }
            ])
        
        return qa_pairs
    
    def create_training_data(self):
        """Create complete training dataset"""
        print("="*60)
        print("CREATING TRAINING DATASET")
        print("="*60)
        
        training_data = []
        
        for sample in tqdm(self.samples, desc="Processing samples"):
            # Generate captions
            captions = self.generate_spectral_captions(sample)
            
            # Generate QA pairs
            qa_pairs = self.generate_vqa_pairs(sample)
            
            # Create training example
            training_example = {
                'sample_id': sample['id'],
                'land_cover': sample.get('land_cover', 'unknown'),
                'spectral_indices': sample.get('spectral_indices', {}),
                'bands': sample.get('bands', {}),
                'captions': captions,
                'qa_pairs': qa_pairs
            }
            
            training_data.append(training_example)
        
        # Save training data
        output_file = self.output_dir / 'training_data.json'
        with open(output_file, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        print(f"\n✓ Created {len(training_data)} training examples")
        print(f"  Saved to: {output_file}")
        
        # Statistics
        total_captions = sum(len(ex['captions']) for ex in training_data)
        total_qa = sum(len(ex['qa_pairs']) for ex in training_data)
        
        print(f"\nDataset Statistics:")
        print(f"  Total samples: {len(training_data)}")
        print(f"  Total captions: {total_captions}")
        print(f"  Total QA pairs: {total_qa}")
        print(f"  Avg captions/sample: {total_captions/len(training_data):.1f}")
        print(f"  Avg QA/sample: {total_qa/len(training_data):.1f}")
        
        # Show example
        print("\n" + "="*60)
        print("EXAMPLE TRAINING SAMPLE")
        print("="*60)
        
        example = training_data[0]
        print(f"\nLand Cover: {example['land_cover']}")
        print(f"NDVI: {example['spectral_indices']['NDVI']:.3f}")
        print(f"NDWI: {example['spectral_indices']['NDWI']:.3f}")
        print(f"NDBI: {example['spectral_indices']['NDBI']:.3f}")
        
        print(f"\nCaption 1:")
        print(f"  {example['captions'][0]}")
        
        print(f"\nSample QA Pair:")
        print(f"  Q: {example['qa_pairs'][0]['question']}")
        print(f"  A: {example['qa_pairs'][0]['answer']}")
        
        return training_data


def main():
    """Create training dataset"""
    
    creator = SpectralDatasetCreator()
    training_data = creator.create_training_data()
    
    print("\n" + "="*60)
    print("DATASET CREATION COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Review training_data.json")
    print("2. Augment dataset if needed (rotation, flipping)")
    print("3. Create PyTorch Dataset class")
    print("4. Begin training!")


if __name__ == "__main__":
    main()