/**
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 *  This header file contains structures, type definitions and function declarations
 *  for using the SMBus driver as a raw i2c driver.
 *
 *  @file i2c.h
 *
 *  @note   [1] This provides a simplified API to use the smbus 3.2 library as a simple
 *              i2c device. Only the most essential commands are provided.
 *  @note   [2] xInitSMBus must still be called to initialise the Bus control.
 *
 */

#ifndef _I2C_H_
#define _I2C_H_

#ifdef __cplusplus
extern "C"
{
#endif

/******************************************************************************/
/* includes                                                                   */
/******************************************************************************/

#include <stdint.h>



/******************************************************************************/
/* defines                                                                    */
/******************************************************************************/

#define I2C_SUCCESS             (   0 )
#define I2C_ERROR               (  -1 )
#define I2C_MAX_BUFFER_LEN      ( 256 )
#define I2C_READ_DATA_SIZE_MIN  (   1 )


/******************************************************************************/
/* enums                                                                      */
/******************************************************************************/

/******************************************************************************/
/* structs                                                                    */
/******************************************************************************/

/*
 * @struct I2C_PROFILE_TYPE
 * @brief  Forward Declaration of structure to hold a single i2c profile
 */
struct I2C_PROFILE_TYPE;

/******************************************************************************/
/* typedefs                                                                   */
/******************************************************************************/

/*
 * @typedef I2C_USER_SUPPLIED_ENVIRONMENT_GET_DATA_TYPE
 * @brief   This callback updates the initialiser with new data
 *
 * @param   pucData     pointer to the new data read
 * @param   pusDataSize     number of bytes in pucData
 *
 * @return  void
 */
typedef void ( *I2C_USER_SUPPLIED_ENVIRONMENT_GET_DATA_TYPE )( uint8_t* pucData, uint16_t* pusDataSize );

/*
 * @typedef I2C_USER_SUPPLIED_ENVIRONMENT_WRITE_DATA_TYPE
 * @brief   This callback retrieves data from the initialiser to write
 *
 * @param   pucData     pointer to the new data to write
 * @param   usDataSize      number of bytes in pucData
 *
 * @return  void
 */
typedef void ( *I2C_USER_SUPPLIED_ENVIRONMENT_WRITE_DATA_TYPE )( uint8_t* pucData, uint16_t usDataSize );

/*
 * @typedef I2C_USER_SUPPLIED_ENVIRONMENT_COMMAND_COMPLETE
 * @brief   This callback updates the initialiser when a command is complete
 *
 * @param   ulStatus is the status of the command
 *
 * @return  void
 */
typedef void ( *I2C_USER_SUPPLIED_ENVIRONMENT_COMMAND_COMPLETE )( uint32_t ulStatus );

/*
 * @typedef I2C_USER_SUPPLIED_ENVIRONMENT_BUS_ERROR
 * @brief   This callback updates the initialiser when there is an i2 Error
 *
 * @param   ucError is the error that was raised
 *
 * @return  void
 */
typedef void ( *I2C_USER_SUPPLIED_ENVIRONMENT_BUS_ERROR )( uint8_t ucError );

/*
 * @typedef I2C_USER_SUPPLIED_ENVIRONMENT_BUS_WARNING
 *
 * @brief   This callback updates the initialiser when there is an i2c Warning
 *
 * @param   ucWarning is the warning that was raised
 *
 * @return  void
 */
typedef void ( *I2C_USER_SUPPLIED_ENVIRONMENT_BUS_WARNING )( uint8_t ucWarning );

/******************************************************************************/
/* function declarations                                                      */
/******************************************************************************/

/******************************************************************************/
/*!
 *  @brief  Creates an i2c device to act as both a master and a slave
 *
 *  @param  pxI2cProfile        Handler to the SMBus profile structure (see smbus.h)
 *  @param  ucAddr              Slave address that this device will respond to.
 *  @param  pFnGetData          Callback called when new data arrives at the device.
 *  @param  pFnWriteData        Callback called when the device has to respond with data.
 *  @param  pFnAnnounceResult   Callback called when a write/read command is complete.
 *  @param  pFnBusError         Callback called when an error occurs
 *
 *  @return The ID of the device (0 to 6)
 *          Returns I2C_ERROR if the call is unsuccessful
 *
 *  @note   [1] This device will be created along with any SMBus instances;
 *              therefore, it requires availability in the SMBus instance pool.
 *              See ucCreateSMBusInstance() in smbus.h for more information.
 *              xInitSMBus() must have been successfully called before this.
 */
/******************************************************************************/
uint8_t ucI2CCreateDevice( struct I2C_PROFILE_TYPE* pxI2cProfile,
                            uint8_t ucAddr,
                            I2C_USER_SUPPLIED_ENVIRONMENT_GET_DATA_TYPE     pFnGetData,
                            I2C_USER_SUPPLIED_ENVIRONMENT_WRITE_DATA_TYPE   pFnWriteData,
                            I2C_USER_SUPPLIED_ENVIRONMENT_COMMAND_COMPLETE  pFnAnnounceResult,
                            I2C_USER_SUPPLIED_ENVIRONMENT_BUS_ERROR         pFnBusError,
                            I2C_USER_SUPPLIED_ENVIRONMENT_BUS_WARNING       pFnBusWarning );
            

/******************************************************************************/
/*!
 *  @brief  Destroys a previously created i2c device
 *
 *  @param  pxI2cProfile    Handler to the SMBus profile structure (see smbus.h)
 *  @param  ucDeviceId      ID of the device to destroy
 *
 *  @return I2C_SUCCESS - the device has been successfully destroyed
 *          I2C_ERROR   - the device has not been destroyed
 *
 *  @note   None
 */
/******************************************************************************/            
uint8_t ucI2CDestroyDevice( struct I2C_PROFILE_TYPE* pxI2cProfile,
                            uint8_t ucDeviceId );


/******************************************************************************/
/*!
 *  @brief  Writes data to a remote slave as a master
 *
 *  @param  pxI2cProfile    Handler to the i2c profile structure
 *  @param  ucDeviceId      Device to use as a master
 *  @param  ucAddr          Address of remote slave to write to
 *  @param  pucData         Data buffer to write (must be at least usNumBytes bytes)
 *  @param  usNumBytes      Number of bytes to write
 *
 *  @return I2C_SUCCESS - data successfully written
 *          I2C_ERROR   - an error occurred attempting to write
 *
 */
/******************************************************************************/    
uint8_t ucI2CWriteData( struct I2C_PROFILE_TYPE* pxI2cProfile,
                        uint8_t  ucDeviceId,
                        uint8_t  ucAddr,
                        uint8_t* pucData,
                        uint16_t usNumBytes );

/******************************************************************************/
/*!
 *  @brief  Reads data from a remote slave as a master
 *
 *  @param  pxI2cProfile    Handler to the i2c profile structure
 *  @param  ucDeviceId      Device to use as a master
 *  @param  ucAddr          Address of remote slave to read from
 *  @param  pusNumBytes     Maximum number of bytes to read
 *                          - Cannot be larger than I2C_MAX_BUFFER_LEN
 *
 *  @return I2C_SUCCESS - data successfully read
 *          I2C_ERROR   - an error occurred attempting to read
 *
 */
/******************************************************************************/            
uint8_t ucI2CReadData( struct I2C_PROFILE_TYPE* pxI2cProfile,
                        uint8_t   ucDeviceId,
                        uint8_t   ucAddr,
                        uint16_t  usNumBytes );

/******************************************************************************/
/*!
 *  @brief  Writes data to a remote slave as a master and then reads from it
 *
 *  @param  pxI2cProfile    Handler to the i2c profile structure
 *  @param  ucDeviceId      Device to use as a master
 *  @param  ucAddr          Address of remote slave to write to
 *  @param  pucWriteData    Data buffer to write (must be at least usNumBytes bytes)
 *  @param  usNumWriteBytes Number of bytes to write
 *  @param  usNumReadBytes  Maximum number of bytes to read
 *
 *  @return I2C_SUCCESS - data successfully written
 *          I2C_ERROR   - an error occurred attempting to write
 *
 */
/******************************************************************************/
uint8_t ucI2CWriteReadData( struct I2C_PROFILE_TYPE* pxI2cProfile,
                            uint8_t   ucDeviceId,
                            uint8_t   ucAddr,
                            uint8_t*  pucWriteData,
                            uint16_t  usNumWriteBytes,
                            uint16_t  usNumReadBytes );

#endif /* _I2C_H_ */
