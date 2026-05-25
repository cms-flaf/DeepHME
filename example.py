import uproot
import awkward as ak
import matplotlib.pyplot as plt
import numpy as np

from src.DeepHME import DeepHME

def main():
    ch = 'DL'
    file = uproot.open('data/dl.root')
    tree = file['Events']
    branches_to_load = ['centralJet_pt', 'centralJet_eta', 'centralJet_phi', 'centralJet_mass',
                        'centralJet_btagPNetB', 'centralJet_btagPNetQvG',
                        'SelectedFatJet_pt', 'SelectedFatJet_eta', 'SelectedFatJet_phi', 'SelectedFatJet_mass',
                        'SelectedFatJet_particleNetWithMass_HbbvsQCD',
                        'lep1_pt', 'lep1_eta', 'lep1_phi', 'lep1_mass',
                        'lep2_pt', 'lep2_eta', 'lep2_phi', 'lep2_mass',
                        'PuppiMET_pt', 'PuppiMET_phi',
                        'event']
    branches = tree.arrays(branches_to_load)
    
    estimator = DeepHME(
        model_name='predict_quantiles3D_DL_v12', 
        channel=ch, 
        return_errors=True,
        met_name='PuppiMET'
    )
    mass, errors = estimator.predict(
        event_id=branches['event'],
        lep1_pt=branches['lep1_pt'], 
        lep1_eta=branches['lep1_eta'], 
        lep1_phi=branches['lep1_phi'], 
        lep1_mass=branches['lep1_mass'],
        lep2_pt=branches['lep2_pt'], 
        lep2_eta=branches['lep2_eta'], 
        lep2_phi=branches['lep2_phi'], 
        lep2_mass=branches['lep2_mass'],
        met_pt=branches['PuppiMET_pt'], 
        met_phi=branches['PuppiMET_phi'],
        jet_pt=branches['centralJet_pt'], 
        jet_eta=branches['centralJet_eta'], 
        jet_phi=branches['centralJet_phi'], 
        jet_mass=branches['centralJet_mass'], 
        jet_btagPNetB=branches['centralJet_btagPNetB'], 
        jet_btagPNetQvG=branches['centralJet_btagPNetQvG'],
        fatjet_pt=branches['SelectedFatJet_pt'], 
        fatjet_eta=branches['SelectedFatJet_eta'], 
        fatjet_phi=branches['SelectedFatJet_phi'], 
        fatjet_mass=branches['SelectedFatJet_mass'],
        fatjet_particleNetWithMass_HbbvsQCD=branches['SelectedFatJet_particleNetWithMass_HbbvsQCD'], 
        output_format='mass'
    )

    plt.hist(mass, bins=np.linspace(0, 2500, 100))
    plt.title('Predicted mass')
    plt.ylabel('Count')
    plt.xlabel('Mass')
    plt.grid(True)
    plt.savefig(f'example_mass_{ch}.pdf', bbox_inches='tight')
    plt.clf()
    plt.close()

    plt.hist(errors, bins=np.linspace(0, 500, 100))
    plt.title('Predicted errors')
    plt.ylabel('Count')
    plt.xlabel('Error')
    plt.grid(True)
    plt.savefig(f'example_errors_{ch}.pdf', bbox_inches='tight')
    plt.clf()
    plt.close()

if __name__ == '__main__':
    main()