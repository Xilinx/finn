/**
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * This file contains the function declarations for the action functions
 *
 * @file smbus_action.h
 *
 */

#ifndef _SMBUS_ACTION_H_
#define _SMBUS_ACTION_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include "smbus.h"

#define A_SMBUS_NO_ACTION                           ( 0x00000000 )
#define A_SMBUS_GET_PROTOCOL                        ( 0x00000001 )
#define A_SMBUS_CREATE_EVENT_SEND_NEXT_BYTE         ( 0x00000002 )
#define MAX_SMBUS_ACTIONS                           ( 3 )
#define SMBUS_ARP_DIRECTED_COMMAND_MASK             ( 0x01 )
#define SMBUS_ARP_UDID_DIRECTED_COMMAND             ( 0x01 )
#define SMBUS_ARP_DIRECTED_COMMAND_ADDRESS_MASK     ( 0xFE )
#define SMBUS_ACTION_ARP_PROTOCOL_DETERMINED        ( 0 )
#define SMBUS_ACTION_ARP_PROTOCOL_UNDETERMINED      ( 1 )

/******************************************************************************
*
* @brief    Clears all actions that have been raised against the specified instance.
*
* @param    pxSMBusInstance is a pointer to the SMBus instance structure.
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
void vSMBusClearAction( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/******************************************************************************
*
* @brief    Sets a specific actions against the specified instance
*
* @param    pxSMBusInstance is a pointer to the SMBus instance structure.
* @param    ulAnyAction is the new action requested against the specified instance.
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
void vSMBusAction( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint32_t ulAnyAction );

/******************************************************************************
*
* @brief    Resets all variables used during an SMBus message transaction back 
*           to their default values
*           Resets the IP's Descriptor and RX FIFOs
*
* @param    pxSMBusInstance is a pointer to the SMBus instance structure.
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
void vSMBusHandleActionResetAllData( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/******************************************************************************
*
* @brief    Generates an E_IS_PEC_REQUIRED event against the specified instance
*
* @param    pxSMBusInstance is a pointer to the SMBus instance structure.
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
void vSMBusHandleActionCreateEventIsPECRequired( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/******************************************************************************
*
* @brief    Generates an E_SEND_NEXT_BYTE event against the specified instance
*
* @param    pxSMBusInstance is a pointer to the SMBus instance structure.
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
void vSMBusHandleActionCreateEventSendNextByte( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/******************************************************************************
*
* @brief    Reads the command byte from the Target RX FIFO, calls the callback function
*           for the specified instance to get the SMBus protocol associated with the 
*           command byte and stores that protocol in the instance structure
*
* @param    pxSMBusInstance is a pointer to the SMBus instance structure.
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
void vSMBusHandleActionGetProtocol( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/******************************************************************************
*
* @brief    Reads the command byte from the Target RX FIFO for the ARP instance,
*           determines the ARP protocol associated with the command byte,
*           stores that protocol in the instance structure
*
* @param    pxSMBusInstance is a pointer to the SMBus instance structure.
*
* @return   SMBUS_ACTION_ARP_PROTOCOL_UNDETERMINED  - If protocol could not be determined
*           SMBUS_ACTION_ARP_PROTOCOL_DETERMINED    - If protocol was able to be determined
*
* @note     None.
*
*****************************************************************************/
uint8_t ucSMBusHandleActionGetARPProtocol( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/******************************************************************************
*
* @brief    Checks if a callback function to read data from the application software 
*           for the specified instance is present.  
*           If present it is called for the current SMBus command.
*           The callback function must return the data and size of data for the command.
*
* @param    pxSMBusInstance is a pointer to the SMBus instance structure.
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
void vSMBusHandleActionGetDataFromApplication( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/******************************************************************************
*
* @brief    Checks if a callback function to write data to the application software 
*           for the specified instance is present.  
*           If present it is called for the current SMBus command.
*           The callback function write the command, data and size of data and transaction ID.
*
* @param    pxSMBusInstance is a pointer to the SMBus instance structure.
* @param    ulTransactionID is current Controller transaction number.
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
void vSMBusHandleActionWriteDataToApplication( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucTransactionID );

/******************************************************************************
*
* @brief    Checks if a callback function to announce the result of the SMBus transaction 
*           for the specified instance is present.  
*           If present the callback is called.
*           The callback function will include the command, transaction ID and the result
*
* @param    pxSMBusInstance is a pointer to the SMBus instance structure.
* @param    ucTransactionID is current Controller transaction number.
* @param    ulStatus is the result of the SMBus transaction
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
void vSMBusHandleActionAnnounceResultToApplication( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucTransactionID, 
                                                        uint32_t ulStatus );

/******************************************************************************
*
* @brief    Checks if a callback function to announce that an ARP Assign Address 
*           for the specified instance is present.  
*           If present the callback is called.
*           The callback function will include the newly assigned address
*
* @param    pxSMBusInstance is a pointer to the SMBus instance structure.  
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
void vSMBusHandleActionNotifyAddressChangeToApplication( SMBUS_INSTANCE_TYPE* pxSMBusInstance, 
                                                                uint8_t ucTransactionID );

/******************************************************************************
*
* @brief    Checks if a callback function to announce an SMBuss Error
*           for the specified instance is present.  
*           If present the callback is called.
*           The callback function will include the error type
*
* @param    pxSMBusInstance is a pointer to the SMBus instance structure.
* @param    ucError is the error enum
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
void vSMBusHandleActionBusError( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucError );

/******************************************************************************
*
* @brief    Checks if a callback function to announce an SMBuss Warning
*           for the specified instance is present.  
*           If present the callback is called.
*           The callback function will include the warning type
*
* @param    pxSMBusInstance is a pointer to the SMBus instance structure.
* @param    ucWarning is the warning enum
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
void vSMBusHandleActionBusWarning( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucWarning );

/******************************************************************************
*
* @brief    Checks if a callback function to read data from the application software 
*           for the specified instance is present.  
*           If present it is called for the current I2C transaction.
*           The callback function must return the data and size of data for the command.
*
* @param    pxSMBusInstance is a pointer to the SMBus instance structure.
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
void vSMBusHandleActionGetI2CDataFromApplication( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/******************************************************************************
*
* @brief    Checks if a callback function to write data to the application software 
*           for the specified instance is present.  
*           If present it is called for the current I2C transaction.
*           The callback function write the command, data and size of data and transaction ID.
*
* @param    pxSMBusInstance is a pointer to the SMBus instance structure.
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
void vSMBusHandleActionWriteI2CDataToApplication( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/******************************************************************************
*
* @brief    Checks if a callback function to announce the result of the I2C transaction 
*           for the specified instance is present.  
*           If present the callback is called.
*           The callback function will include the command and the result
*
* @param    pxSMBusInstance is a pointer to the SMBus instance structure.
* @param    ulStatus is the result of the SMBus transaction
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
void vSMBusHandleActionAnnounceI2CResultToApplication( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint32_t ulStatus );


#ifdef __cplusplus
}
#endif

#endif /* _SMBUS_ACTION_H_ */
