// Classification dataset .JSON example
// ------------------------------------
// {
//     "samples": [
//         {
//             "filename": "tesco_1.jpg",
//             "segments": [
//                 {
//                     "classification": "T",
//                     "meta": {
//                         "no": 1
//                     },
//                     "values": [
//                         {
//                             "type": "hu1",
//                             "value": 123456789
//                         },
//                         {
//                             "type": "hu2",
//                             "value": 123456789
//                         },
//                         {
//                             "type": "hu3",
//                             "value": 123456789
//                         },
//                         {
//                             "type": "hu4",
//                             "value": 123456789
//                         },
//                         {
//                             "type": "hu5",
//                             "value": 123456789
//                         },
//                         {
//                             "type": "hu6",
//                             "value": 123456789
//                         },
//                         {
//                             "type": "hu7",
//                             "value": 123456789
//                         }
//                     ]
//                 }
//             ]
//         }
//     ]
// }

function calculateRanges(samples) {
    const ranges = {};

    samples.forEach((sample) => {
        sample.segments.forEach((segment) => {
            const classification = segment.classification;
            const values = segment.values;

            if (ranges[classification]) {
                return;
            }

            ranges[classification] = {
                values: {}
            };

            values.forEach((valueObj) => {
                ranges[classification].values[valueObj.type] = {
                    min: Infinity,
                    max: -Infinity,
                    values: []
                };
            });
        });
    });

    samples.forEach((sample) => {
        sample.segments.forEach((segment) => {
            const classification = segment.classification;
            const values = segment.values;
            const classEntry = ranges[classification];

            values.forEach((valueObj) => {
                const rangeEntry = classEntry.values[valueObj.type];
                const curMin = rangeEntry.min;
                const curMax = rangeEntry.max;

                if (curMin > valueObj.value) {
                    rangeEntry.min = valueObj.value;
                }
                if (curMax < valueObj.value) {
                    rangeEntry.max = valueObj.value;
                }

                rangeEntry.values.push(valueObj.value);
            });
        });
    });

    return ranges;
}
