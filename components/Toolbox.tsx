import React, { useState, useCallback, useRef } from 'react';
import { TOOLBOX_MODULES } from '../constants';
import { ModuleType } from '../types';
import { LinkIcon, ChevronUpIcon, ChevronDownIcon } from './icons';

interface ToolboxProps {
    onModuleDoubleClick: (type: ModuleType) => void;
}

const preprocessTypes = [
    ModuleType.LoadData,
    ModuleType.Statistics,
    ModuleType.SelectData,
    ModuleType.HandleMissingValues,
    ModuleType.TransformData,
    ModuleType.EncodeCategorical,
    ModuleType.NormalizeData,
    ModuleType.TransitionData,
    ModuleType.ResampleData,
];

const analysisOpTypes = [
    ModuleType.SplitData,
    ModuleType.TrainModel,
    ModuleType.ScoreModel,
    ModuleType.EvaluateModel,
];

const supervisedLearningTypes = [
    ModuleType.LinearRegression,
    ModuleType.LogisticRegression,
    ModuleType.PoissonRegression,
    ModuleType.NegativeBinomialRegression,
    ModuleType.DecisionTree,
    ModuleType.RandomForest,
    ModuleType.SVM,
    ModuleType.LinearDiscriminantAnalysis,
    ModuleType.NaiveBayes,
    ModuleType.KNN,
];

const unsupervisedModelTypes = [
    ModuleType.KMeans,
    ModuleType.HierarchicalClustering,
    ModuleType.DBSCAN,
    ModuleType.PrincipalComponentAnalysis,
];

const traditionAnalysisTypes = [
    ModuleType.StatModels,
    ModuleType.ResultModel,
    ModuleType.PredictModel,
];

const reinsuranceTypes = [
    ModuleType.FitLossDistribution,
    ModuleType.GenerateExposureCurve,
    ModuleType.PriceXoLLayer,
];

const xolPricingTypes = [
    ModuleType.XolLoading,
    ModuleType.ApplyThreshold,
    ModuleType.DefineXolContract,
    ModuleType.CalculateCededLoss,
    ModuleType.PriceXolContract,
];

const categorizedModules = [
    {
        name: 'Data Preprocess',
        modules: TOOLBOX_MODULES.filter(m => preprocessTypes.includes(m.type))
    },
    {
        name: 'Data Analysis',
        subCategories: [
            { name: 'Operations', modules: TOOLBOX_MODULES.filter(m => analysisOpTypes.includes(m.type)) },
            { name: 'Supervised Learning', modules: TOOLBOX_MODULES.filter(m => supervisedLearningTypes.includes(m.type)) },
            { name: 'Unsupervised Learning', modules: TOOLBOX_MODULES.filter(m => unsupervisedModelTypes.includes(m.type)) },
        ]
    },
    {
        name: 'Tradition Analysis',
        modules: TOOLBOX_MODULES.filter(m => traditionAnalysisTypes.includes(m.type))
    },
    {
        name: 'Reinsurance Analysis',
        modules: TOOLBOX_MODULES.filter(m => reinsuranceTypes.includes(m.type))
    },
    {
        name: 'XoL Reinsurance Pricing',
        modules: TOOLBOX_MODULES.filter(m => xolPricingTypes.includes(m.type))
    }
];

const ToolboxItem: React.FC<{ type: ModuleType, name: string, icon: React.FC<any>, description: string, onDoubleClick: (type: ModuleType) => void, onTouchEnd: (type: ModuleType, e: React.TouchEvent) => void }> = 
({ type, name, icon: Icon, description, onDoubleClick, onTouchEnd }) => {
    const handleDragStart = (e: React.DragEvent<HTMLDivElement>, type: ModuleType) => {
      e.dataTransfer.setData('application/reactflow', type);
      e.dataTransfer.effectAllowed = 'copy'; // Changed from 'move' to 'copy' for better UX when creating new items
    };

    return (
        <div
            onDragStart={(e) => handleDragStart(e, type)}
            onDoubleClick={() => onDoubleClick(type)}
            onTouchEnd={(e) => onTouchEnd(type, e)}
            draggable
            title={description}
            className="flex items-center px-3 py-2 rounded-lg cursor-grab bg-gray-800 hover:bg-gray-700 hover:text-blue-400 transition-colors"
        >
            <Icon className="h-5 w-5 mr-3 flex-shrink-0" />
            <span className="text-sm font-medium">{name}</span>
        </div>
    );
};


export const Toolbox: React.FC<ToolboxProps> = ({ onModuleDoubleClick }) => {
    const [expandedCategories, setExpandedCategories] = useState<Record<string, boolean>>({
        'Data Preprocess': true,
        'Data Analysis': true,
        'Tradition Analysis': true,
        'Reinsurance Analysis': true,
        'XoL Reinsurance Pricing': true,
        'Operations': true,
        'Supervised Learning': true,
        'Unsupervised Learning': true,
    });

    const [lastTapInfo, setLastTapInfo] = useState<{ type: ModuleType; time: number } | null>(null);

    const toggleCategory = (categoryName: string) => {
        setExpandedCategories(prev => ({
            ...prev,
            [categoryName]: !prev[categoryName]
        }));
    };

    const handleTouchEnd = useCallback((type: ModuleType, e: React.TouchEvent) => {
        const now = Date.now();
        const DOUBLE_TAP_DELAY = 300; // ms

        if (lastTapInfo && lastTapInfo.type === type && (now - lastTapInfo.time < DOUBLE_TAP_DELAY)) {
            // Double tap detected for this module type
            e.preventDefault(); // Prevent default touch behavior (e.g., zoom)
            onModuleDoubleClick(type);
            setLastTapInfo(null); // Reset
        } else {
            setLastTapInfo({ type, time: now });
        }
    }, [lastTapInfo, onModuleDoubleClick]);

  return (
    <aside className="w-56 bg-gray-900 border-r border-gray-700 flex flex-col h-full">
        <div className="p-3 border-b border-gray-700 flex-shrink-0">
            <h3 className="text-lg font-semibold text-white">Modules</h3>
        </div>
        <div className="flex-1 p-2 overflow-y-auto panel-scrollbar min-h-0">
            <div className="flex flex-col gap-2">
            {categorizedModules.map(category => (
                <div key={category.name}>
                    <button
                        onClick={() => toggleCategory(category.name)}
                        className="w-full flex items-center justify-between p-2 rounded-lg text-left text-sm font-semibold text-gray-300 hover:bg-gray-800 transition-colors"
                    >
                        <span>{category.name}</span>
                        {expandedCategories[category.name] ? <ChevronUpIcon className="w-5 h-5" /> : <ChevronDownIcon className="w-5 h-5" />}
                    </button>
                    {expandedCategories[category.name] && (
                        <div className="pl-2 pt-2 flex flex-col gap-2">
                            {category.modules?.map(({ type, name, icon, description }) => (
                                <ToolboxItem key={type} type={type} name={name} icon={icon} description={description} onDoubleClick={onModuleDoubleClick} onTouchEnd={handleTouchEnd} />
                            ))}
                            {category.subCategories?.map(subCategory => (
                                <div key={subCategory.name} className="pl-2">
                                     <button
                                        onClick={() => toggleCategory(subCategory.name)}
                                        className="w-full flex items-center justify-between py-1 rounded-md text-left text-xs font-semibold text-gray-400 hover:bg-gray-800 transition-colors"
                                    >
                                        <span>{subCategory.name}</span>
                                        {expandedCategories[subCategory.name] ? <ChevronUpIcon className="w-4 h-4" /> : <ChevronDownIcon className="w-4 h-4" />}
                                    </button>
                                    {expandedCategories[subCategory.name] && (
                                        <div className="pt-2 flex flex-col gap-2">
                                            {subCategory.modules.map(({ type, name, icon, description }) => (
                                                <ToolboxItem key={type} type={type} name={name} icon={icon} description={description} onDoubleClick={onModuleDoubleClick} onTouchEnd={handleTouchEnd} />
                                            ))}
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            ))}
            </div>
        </div>
        <div className="p-2 border-t border-gray-700 flex-shrink-0">
          <p className="text-sm text-gray-400 text-center mb-2">Developed by TKLEEN</p>
          <a
            href="https://www.ai4insurance.com"
            target="_blank"
            rel="noopener noreferrer"
            title="Go to ai4insurance.com"
            className="mx-auto flex items-center justify-center w-6 h-6 bg-gray-600 hover:bg-gray-500 rounded-md text-white transition-colors"
          >
              <LinkIcon className="w-5 h-5" />
          </a>
      </div>
    </aside>
  );
};